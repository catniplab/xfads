"""
Low-rank nonlinear smoother implementations used across XFADS.

This module provides Torch modules and helper routines for smoothing in
low-rank latent linear dynamical systems with nonlinear emission models.
The implementations support stochastic variational inference, causal and
acausal filtering, and utilities for computing Gaussian update terms that
appear in the variational objective.
"""

import math
import torch
from torch import nn
import torch.nn.functional as Fn
from .. import linalg_utils

from ..utils import pad_mask
from ..linalg_utils import bmv, bip, bop, chol_bmv_solve


class LowRankNonlinearStateSpaceModel(nn.Module):
    """
    Variational smoother for low-rank nonlinear state-space models.

    The module couples local encoder statistics with backward message passing
    and a nonlinear filter to produce smoothed latent trajectories and the
    evidence lower bound (ELBO) contribution for a batch of neural recordings.

    Parameters
    ----------
    dynamics_mod : nn.Module
        Dynamics module that exposes `mean_fn` and a `log_Q` diagonal noise
        parameter of shape ``[L]`` for ``L`` latent dimensions.
    likelihood_pdf : nn.Module
        Likelihood module with ``get_ell`` and ``readout_fn`` interfaces used to
        evaluate emission log-likelihoods.
    initial_c_pdf : nn.Module
        Module describing the initial latent distribution with attributes
        ``m_0`` and ``log_Q_0``.
    backward_encoder : nn.Module
        Encoder generating backward messages ``(k_b, K_b)`` from local encoder
        outputs, each tensor shaped ``[batch, time, latent, rank]``.
    local_encoder : nn.Module
        Encoder mapping observations to canonical statistics ``(k_y, K_y)`` with
        the same shapes as the backward encoder outputs.
    nl_filter : nn.Module
        Nonlinear filter that accepts canonical statistics and returns sampled
        latent trajectories and summary statistics.
    """

    def __init__(
        self,
        dynamics_mod,
        likelihood_pdf,
        initial_c_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
    ):
        """
        Initialize the smoother with model components.

        Parameters
        ----------
        dynamics_mod : nn.Module
            Latent dynamics module providing ``mean_fn`` and ``log_Q``.
        likelihood_pdf : nn.Module
            Emission model supporting ``get_ell`` and ``readout_fn``.
        initial_c_pdf : nn.Module
            Initial latent distribution with ``m_0`` and ``log_Q_0`` tensors.
        backward_encoder : nn.Module
            Module that converts local encoder outputs into backward messages.
        local_encoder : nn.Module
            Module returning canonical statistics from observations.
        nl_filter : nn.Module
            Filtering module producing smoothed samples and statistics.
        """
        super().__init__()

        self.nl_filter = nl_filter
        self.dynamics_mod = dynamics_mod
        self.local_encoder = local_encoder
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf
        self.backward_encoder = backward_encoder

    @torch.jit.export
    def forward(
        self,
        y,
        n_samples: int,
        p_mask_y_in: float = 0.0,
        p_mask_apb: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        get_P_s: bool = False,
    ):
        """
        Compute the ELBO contribution and smoothed latent trajectories.

        Parameters
        ----------
        y : torch.Tensor
            Observations with shape ``[batch, time, neurons]``.
        n_samples : int
            Number of latent samples drawn by the nonlinear filter.
        p_mask_y_in : float, optional
            Dropout probability applied to observation entries, by default 0.0.
        p_mask_apb : float, optional
            Dropout probability for combined encoder statistics, by default 0.0.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        get_P_s : bool, optional
            If True, request smoothed covariance summaries from the filter.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Loss scalar, smoothed latent samples shaped
            ``[n_samples, batch, latents, time]``, and auxiliary statistics.
        """

        z_s, stats = self.fast_smooth_1_to_T(
            y,
            n_samples,
            p_mask_apb=p_mask_apb,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_b=p_mask_b,
            get_kl=True,
            get_P_s=get_P_s,
        )

        ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)
        loss = stats["kl"] - ell
        loss = loss.sum(dim=-1).mean()
        return loss, z_s, stats

    def fast_filter_1_to_T(
        self,
        y,
        n_samples: int,
        p_mask_y_in: float = 0.0,
        p_mask_a: float = 0.0,
        get_kl: bool = False,
        get_v: bool = False,
    ):
        """
        Run the causal filter to obtain latent samples and statistics.

        Parameters
        ----------
        y : torch.Tensor
            Observations of shape ``[batch, time, neurons]``.
        n_samples : int
            Number of Monte Carlo samples produced by the filter.
        p_mask_y_in : float, optional
            Dropout probability applied to the observation tensor, by default 0.0.
        p_mask_a : float, optional
            Dropout probability for encoder statistics, by default 0.0.
        get_kl : bool, optional
            If True, return per-time-step KL divergences, by default False.
        get_v : bool, optional
            If True, request predictive variances from the filter, by default False.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Filtered latent samples with shape
            ``[n_samples, batch, latents, time]`` and a statistics dictionary.
        """
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons))
        )
        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins)))

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        z_s, stats = self.nl_filter(k_y, K_y, n_samples, get_kl=get_kl, get_v=get_v)
        stats["t_mask_y_in"] = t_mask_y_in

        return z_s, stats

    def fast_smooth_1_to_T(
        self,
        y,
        n_samples: int,
        p_mask_a: float = 0.0,
        p_mask_apb: float = 0.0,
        p_mask_y_in: float = 0.0,
        p_mask_b: float = 0.0,
        get_kl: bool = False,
        get_v: bool = False,
        get_P_s: bool = False,
    ):
        """
        Run the acausal smoother across all time steps.

        Parameters
        ----------
        y : torch.Tensor
            Observations with shape ``[batch, time, neurons]``.
        n_samples : int
            Number of samples returned by the nonlinear filter.
        p_mask_a : float, optional
            Dropout probability for local encoder statistics, by default 0.0.
        p_mask_apb : float, optional
            Dropout probability for combined encoder statistics, by default 0.0.
        p_mask_y_in : float, optional
            Dropout probability applied to the observation tensor, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        get_kl : bool, optional
            If True, return KL contributions per time step, by default False.
        get_v : bool, optional
            If True, return predictive variances from the filter, by default False.
        get_P_s : bool, optional
            If True, include smoothed covariance estimates in the stats dict.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Smoothed latent samples shaped ``[n_samples, batch, latents, time]``
            and a statistics dictionary containing encoder masks and KL terms.
        """

        device = y.device
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli(
            (1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=y.device)
        )
        t_mask_b = torch.bernoulli(
            (1 - p_mask_b) * torch.ones((n_trials, n_time_bins), device=y.device)
        )
        t_mask_apb = torch.bernoulli(
            (1 - p_mask_apb) * torch.ones((n_trials, n_time_bins), device=y.device)
        )
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in)
            * torch.ones((n_trials, n_time_bins, n_neurons), device=y.device)
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b
        K_b = t_mask_b[..., None, None] * K_b

        k = k_b + k_y
        K = torch.concat([K_b, K_y], dim=-1)
        k = t_mask_apb[..., None] * k
        K = t_mask_apb[..., None, None] * K

        z_s, stats = self.nl_filter(
            k, K, n_samples, get_kl=get_kl, get_v=get_v, get_P_s=get_P_s
        )
        stats["t_mask_y_in"] = t_mask_y_in

        return z_s, stats

    def predict_forward(self, z_tm1: torch.Tensor, n_bins: int):
        """
        Sample latent trajectories forward in time from the dynamics model.

        Parameters
        ----------
        z_tm1 : torch.Tensor
            Latent state at the previous time step with shape
            ``[batch, latents]``.
        n_bins : int
            Number of future time steps to simulate.

        Returns
        -------
        torch.Tensor
            Simulated latent states with shape ``[batch, latents, n_bins]``.
        """
        z_forward = []
        Q_sqrt = torch.sqrt(Fn.softplus(self.dynamics_mod.log_Q))

        for t in range(n_bins):
            if t == 0:
                z_t = self.dynamics_mod.mean_fn(z_tm1) + Q_sqrt * torch.randn_like(
                    z_tm1
                )
            else:
                z_t = self.dynamics_mod.mean_fn(
                    z_forward[t - 1]
                ) + Q_sqrt * torch.randn_like(z_forward[t - 1])

            z_forward.append(z_t)

        z_forward = torch.stack(z_forward, dim=2)
        return z_forward


class LrSSMcoBPSheldinEncoder(LowRankNonlinearStateSpaceModel):
    """
    Smoother that conditions on an observed subset of neurons during training.

    This encoder uses the held-in neuron set to compute statistics while
    allowing masked dropout during ELBO maximization. It also supports
    predicting held-out neuron firing rates and optionally contrastive
    divergence updates.
    """

    def __init__(
        self,
        dynamics_mod,
        likelihood_pdf,
        initial_c_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
        n_neurons_enc,
        n_neurons_obs,
        n_time_bins_enc,
    ):
        """
        Initialize the encoder with observation subset metadata.

        Parameters
        ----------
        dynamics_mod : nn.Module
            Latent dynamics component.
        likelihood_pdf : nn.Module
            Likelihood model with ``get_ell`` and ``readout_fn`` methods.
        initial_c_pdf : nn.Module
            Initial latent distribution.
        backward_encoder : nn.Module
            Module generating backward canonical statistics.
        local_encoder : nn.Module
            Module generating local canonical statistics.
        nl_filter : nn.Module
            Nonlinear filtering backend.
        n_neurons_enc : int
            Number of neurons observed during encoding.
        n_neurons_obs : int
            Total number of observed neurons.
        n_time_bins_enc : int
            Number of time bins used during encoding.
        """
        super().__init__()

        self.nl_filter = nl_filter
        self.dynamics_mod = dynamics_mod
        self.local_encoder = local_encoder
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf
        self.backward_encoder = backward_encoder

        self.n_neurons_enc = n_neurons_enc
        self.n_neurons_obs = n_neurons_obs
        self.n_time_bins_enc = n_time_bins_enc

    @torch.jit.export
    def forward(
        self,
        y_obs,
        n_samples: int,
        p_mask_apb: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        p_mask_y_in: float = 0.0,
        l2_C: float = 1e-1,
        use_cd=False,
    ):
        """
        Evaluate the ELBO using the held-in encoder subset.

        Parameters
        ----------
        y_obs : torch.Tensor
            Observations with shape ``[batch, time, neurons]`` where the first
            ``n_neurons_enc`` columns correspond to held-in neurons.
        n_samples : int
            Number of latent samples to draw.
        p_mask_apb : float, optional
            Dropout probability for combined encoder statistics, by default 0.0.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        l2_C : float, optional
            Regularization coefficient on the readout weights, by default 1e-1.
        use_cd : bool, optional
            If True, apply contrastive divergence style gradient masking.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Loss scalar, smoothed latent samples, and auxiliary statistics.
        """
        n_trials, _, n_neurons_obs = y_obs.shape

        z_enc, stats = self.fast_smooth_1_to_T(
            y_obs[..., : self.n_time_bins_enc, : self.n_neurons_enc],
            n_samples,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_apb=p_mask_apb,
            p_mask_b=p_mask_b,
            get_kl=True,
        )

        if use_cd:
            ell_enc = self.likelihood_pdf.get_ell(
                y_obs[:, : self.n_time_bins_enc], z_enc, reduce_neuron_dim=False
            ).mean(dim=0)
            grad_mask = pad_mask(stats["t_mask_y_in"], ell_enc, 0.0)

            ell_enc_no_grad = (ell_enc * grad_mask).detach()
            ell_grad = ell_enc * (1 - grad_mask)
            ell_enc = (ell_enc_no_grad + ell_grad).sum(dim=-1)
        else:
            ell_enc = self.likelihood_pdf.get_ell(
                y_obs[:, : self.n_time_bins_enc], z_enc
            ).mean(dim=0)

        C = self.likelihood_pdf.readout_fn[-1].weight
        loss_s = stats["kl"] - ell_enc
        loss_s = loss_s.sum(dim=-1).mean()
        loss_s += l2_C * C.pow(2).sum()
        stats["ell"] = ell_enc

        return loss_s, z_enc, stats

    @torch.jit.export
    def predict(
        self,
        y_enc,
        n_samples: int,
        p_mask_y_in: float = 0.0,
    ):
        """
        Produce smoothed latents and rate predictions from encoder-only data.

        Parameters
        ----------
        y_enc : torch.Tensor
            Encoder observations shaped ``[batch, time, n_neurons_enc]``.
        n_samples : int
            Number of latent samples to draw.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Smoothed latents and statistics including expected log firing rates.
        """
        z_s, stats = self.fast_smooth_1_to_T(y_enc, n_samples, get_kl=True)

        # expected log rate
        log_rate_hat = math.log(
            self.likelihood_pdf.delta
        ) + self.likelihood_pdf.readout_fn(stats["m_f"])
        stats["log_rate"] = log_rate_hat
        return z_s, stats


class LrSSMcoBPSallEncoder(LowRankNonlinearStateSpaceModel):
    """
    Encoder that conditions on all observed neurons while tracking metadata.

    In contrast to the held-in encoder, this variant uses the full observation
    tensor during smoothing, which is useful for semi-supervised fine-tuning or
    ablations that mask subsets at inference time only.
    """

    def __init__(
        self,
        dynamics_mod,
        likelihood_pdf,
        initial_c_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
        n_neurons_enc,
        n_neurons_obs,
        n_time_bins_enc,
    ):
        """
        Initialize the encoder with observation metadata.

        Parameters
        ----------
        dynamics_mod : nn.Module
            Latent dynamics component.
        likelihood_pdf : nn.Module
            Likelihood model for emission probabilities.
        initial_c_pdf : nn.Module
            Initial latent distribution.
        backward_encoder : nn.Module
            Module generating backward statistics.
        local_encoder : nn.Module
            Module generating encoder statistics from observations.
        nl_filter : nn.Module
            Nonlinear filtering module.
        n_neurons_enc : int
            Number of neurons used during encoding.
        n_neurons_obs : int
            Total number of observed neurons.
        n_time_bins_enc : int
            Number of time bins considered during encoding.
        """
        super().__init__()

        self.nl_filter = nl_filter
        self.dynamics_mod = dynamics_mod
        self.local_encoder = local_encoder
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf
        self.backward_encoder = backward_encoder

        self.n_neurons_enc = n_neurons_enc
        self.n_neurons_obs = n_neurons_obs
        self.n_time_bins_enc = n_time_bins_enc

    def fast_smooth_1_to_T(
        self,
        y,
        n_samples: int,
        p_mask_a: float = 0.0,
        p_mask_apb: float = 0.0,
        p_mask_y_in: float = 0.0,
        p_mask_b: float = 0.0,
        get_kl: bool = False,
        get_v: bool = False,
    ):
        """
        Run smoothing with dropout masks while conditioning on all neurons.

        Parameters
        ----------
        y : torch.Tensor
            Observations shaped ``[batch, time, neurons]``.
        n_samples : int
            Number of latent samples to generate.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_apb : float, optional
            Dropout probability for combined statistics, by default 0.0.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward statistics, by default 0.0.
        get_kl : bool, optional
            If True, request KL divergence terms, by default False.
        get_v : bool, optional
            If True, request predictive variances, by default False.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Smoothed latent samples and encoder/filter statistics.
        """
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins)))
        t_mask_b = torch.bernoulli((1 - p_mask_b) * torch.ones((n_trials, n_time_bins)))
        t_mask_apb = torch.bernoulli(
            (1 - p_mask_apb) * torch.ones((n_trials, n_time_bins))
        )
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, 1, n_neurons))
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b
        K_b = t_mask_b[..., None, None] * K_b

        k = k_b + k_y
        K = torch.concat([K_b, K_y], dim=-1)
        k = t_mask_apb[..., None] * k
        K = t_mask_apb[..., None, None] * K

        z_s, stats = self.nl_filter(k, K, n_samples, get_kl=get_kl, get_v=get_v)
        stats["t_mask_y_in"] = t_mask_y_in

        return z_s, stats

    @torch.jit.export
    def forward(
        self,
        y_obs,
        n_samples: int,
        p_mask_apb: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        p_mask_y_in: float = 0.0,
        l2_C: float = 1e-1,
        use_cd=False,
    ):
        """
        Evaluate the ELBO while conditioning on all observed neurons.

        Parameters
        ----------
        y_obs : torch.Tensor
            Observations shaped ``[batch, time, n_neurons_obs]``.
        n_samples : int
            Number of latent samples drawn by the filter.
        p_mask_apb : float, optional
            Dropout probability for combined statistics, by default 0.0.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        l2_C : float, optional
            L2 penalty applied to the readout weights, by default 1e-1.
        use_cd : bool, optional
            Unused flag for API compatibility, by default False.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Loss scalar, smoothed latent samples, and statistics dictionary.
        """
        n_trials, _, n_neurons_obs = y_obs.shape

        z_enc, stats = self.fast_smooth_1_to_T(
            y_obs[..., : self.n_time_bins_enc, :],
            n_samples,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_apb=p_mask_apb,
            p_mask_b=p_mask_b,
            get_kl=True,
        )

        ell_enc = self.likelihood_pdf.get_ell(
            y_obs[:, : self.n_time_bins_enc], z_enc
        ).mean(dim=0)
        C = self.likelihood_pdf.readout_fn[-1].weight
        loss_s = stats["kl"] - ell_enc
        loss_s = loss_s.sum(dim=-1).mean()
        loss_s += l2_C * C.pow(2).sum()
        stats["ell"] = ell_enc

        return loss_s, z_enc, stats

    @torch.jit.export
    def predict(self, y_enc, n_samples: int, p_mask_y_in: float = 0.0):
        """
        Infer latents and predict firing rates from encoder-only observations.

        Parameters
        ----------
        y_enc : torch.Tensor
            Observations limited to encoding neurons.
        n_samples : int
            Number of latent samples to produce.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Smoothed latents and statistics including expected log rates.
        """
        n_neurons_heldout = self.n_neurons_obs - self.n_neurons_enc
        y_heldout = torch.zeros((y_enc.shape[0], y_enc.shape[1], n_neurons_heldout))
        y_input = torch.cat([y_enc / (1 - p_mask_y_in), y_heldout], dim=-1)
        z_s, stats = self.fast_smooth_1_to_T(y_input, n_samples, get_kl=True)

        # expected log rate
        log_rate_hat = math.log(
            self.likelihood_pdf.delta
        ) + self.likelihood_pdf.readout_fn(stats["m_f"])
        stats["log_rate"] = log_rate_hat
        return z_s, stats


class NonlinearFilter(nn.Module):
    """
    Low-rank nonlinear filter that supports variational smoothing updates.

    The filter operates on canonical statistics from encoder modules and
    produces latent samples, posterior means, and KL contributions needed by
    the ELBO objective.
    """

    def __init__(self, dynamics_mod, initial_c_pdf):
        """
        Parameters
        ----------
        dynamics_mod : nn.Module
            Latent dynamics module providing ``mean_fn`` and ``log_Q``.
        initial_c_pdf : nn.Module
            Initial latent distribution with ``m_0`` and ``log_Q_0`` tensors.
        """
        super().__init__()

        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf

    def forward(
        self,
        k: torch.Tensor,
        K: torch.Tensor,
        n_samples: int,
        get_v: bool = False,
        get_kl: bool = False,
        p_mask: float = 0.0,
        get_P_s: bool = False,
    ):
        """
        Perform filtering given canonical statistics.

        Parameters
        ----------
        k : torch.Tensor
            Linear canonical statistics of shape ``[batch, time, latents]``.
        K : torch.Tensor
            Quadratic canonical statistics with shape
            ``[batch, time, latents, rank]``.
        n_samples : int
            Number of Monte Carlo samples to draw.
        get_v : bool, optional
            Unused placeholder for variance requests, by default False.
        get_kl : bool, optional
            If True, compute KL divergence terms, by default False.
        p_mask : float, optional
            Dropout probability applied to statistics (unused), by default 0.0.
        get_P_s : bool, optional
            If True, accumulate smoothing covariance estimates.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Filtered latent samples and statistics including posterior means,
            KL divergence, and optional covariances.
        """
        # mask data, 0: data available, 1: data missing
        n_trials, n_time_bins, n_latents, rank = K.shape

        kl = []
        m_f = []
        z_f = []
        stats = {}

        if get_P_s:
            P_s = []

        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        Q_sqrt_diag = torch.sqrt(Q_diag)
        Q_inv_diag = 1 / Q_diag

        for t in range(n_time_bins):
            if t == 0:
                m_0 = self.initial_c_pdf.m_0
                P_p_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
                z_f_t, m_f_t, m_p_t, Psi_f_t, P_p_diag = fast_filter_step_0(
                    m_0, k[:, 0], K[:, 0], P_p_diag, n_samples
                )

                if get_kl:
                    kl.append(
                        low_rank_kl_step_0(m_f_t, m_p_t, P_p_diag, K[:, 0], Psi_f_t)
                    )

                if get_P_s:
                    P_s.append(get_P_s_1(P_p_diag, K[:, 0]))

            else:
                m_fn_z_tm1 = self.dynamics_mod.mean_fn(z_f[t - 1]).movedim(0, -1)
                z_f_t, m_f_t, m_p_t, M_p_c_t, Psi_f_t, Psi_p_t = fast_filter_step_t(
                    m_fn_z_tm1, k[:, t], K[:, t], Q_diag, torch.tensor(False)
                )

                if get_kl:
                    kl.append(
                        low_rank_kl_step_t(
                            m_f_t,
                            m_p_t,
                            M_p_c_t,
                            K[:, t],
                            Psi_f_t,
                            Psi_p_t,
                            Q_inv_diag,
                            Q_sqrt_diag,
                        )
                    )

                if get_P_s:
                    P_s.append(get_P_s_t(Q_diag, M_p_c_t, K[:, t]))

            m_f.append(m_f_t)
            z_f.append(z_f_t)

        z_f = torch.stack(z_f, dim=2)
        stats["m_f"] = torch.stack(m_f, dim=1)

        if get_kl:
            stats["kl"] = torch.stack(kl, dim=1)

        if get_P_s:
            stats["P_s"] = torch.stack(P_s, dim=1)

        return z_f, stats


class NonlinearFilterSmallL(nn.Module):
    """
    Full-rank nonlinear filter for small latent dimensionalities.

    This variant maintains full covariance Cholesky factors instead of the
    low-rank parameterization used in :class:`NonlinearFilter`.
    """

    def __init__(self, dynamics_mod, initial_c_pdf):
        """
        Parameters
        ----------
        dynamics_mod : nn.Module
            Latent dynamics module providing ``mean_fn`` and ``log_Q``.
        initial_c_pdf : nn.Module
            Initial latent distribution with ``m_0`` and ``log_Q_0`` tensors.
        """
        super().__init__()

        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf

    def forward(
        self,
        k: torch.Tensor,
        K: torch.Tensor,
        n_samples: int,
        get_v: bool = False,
        get_kl: bool = False,
        p_mask: float = 0.0,
        get_P_s: bool = False,
    ):
        """
        Perform filtering using full-rank covariance factors.

        Parameters
        ----------
        k : torch.Tensor
            Linear canonical statistics of shape ``[batch, time, latents]``.
        K : torch.Tensor
            Quadratic canonical statistics with shape
            ``[batch, time, latents, rank]``.
        n_samples : int
            Number of latent samples to draw.
        get_v : bool, optional
            Unused placeholder for predictive variance outputs, by default False.
        get_kl : bool, optional
            If True, compute KL divergence terms, by default False.
        p_mask : float, optional
            Unused dropout probability for compatibility, by default 0.0.
        get_P_s : bool, optional
            Unused argument for compatibility with the low-rank filter.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Filtered latent samples and statistics dict containing posterior
            means, predictions, and KL divergence terms.
        """

        # mask data, 0: data available, 1: data missing
        n_trials, n_time_bins, n_latents, rank = K.shape
        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        t_mask = torch.rand(n_time_bins) < p_mask

        z_f = []
        m_p = []
        m_f = []
        P_p_chol = []
        P_f_chol = []
        stats = {}

        for t in range(n_time_bins):
            if t == 0:
                m_0 = self.initial_c_pdf.m_0
                P_0_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)

                z_f_t, m_f_t, P_f_chol_t, P_p_chol_t = filter_step_0(
                    m_0, k[:, 0], K[:, 0], P_0_diag, n_samples
                )
                m_p.append(m_0 * torch.ones(n_trials, n_latents, device=m_0.device))
            else:
                m_fn_z_tm1 = self.dynamics_mod.mean_fn(z_f[t - 1]).movedim(0, -1)
                z_f_t, m_p_t, m_f_t, P_f_chol_t, P_p_chol_t = filter_step_t(
                    m_fn_z_tm1, k[:, t], K[:, t], Q_diag, t_mask[t]
                )
                m_p.append(m_p_t)

            z_f.append(z_f_t)
            m_f.append(m_f_t)
            P_f_chol.append(P_f_chol_t)
            P_p_chol.append(P_p_chol_t)

        z_f = torch.stack(z_f, dim=2)
        stats["m_f"] = torch.stack(m_f, dim=1)
        stats["m_p"] = torch.stack(m_p, dim=1)
        stats["P_f_chol"] = torch.stack(P_f_chol, dim=1)
        stats["P_p_chol"] = torch.stack(P_p_chol, dim=1)

        kl = full_rank_mvn_kl(
            stats["m_f"], stats["P_f_chol"], stats["m_p"], stats["P_p_chol"]
        )
        stats["kl"] = kl

        return z_f, stats


def get_P_s_t(Q_diag, M_p_c_t, K):
    """
    Compute the smoothed covariance at time ``t`` for the low-rank filter.

    Parameters
    ----------
    Q_diag : torch.Tensor
        Diagonal process noise with shape ``[latents]``.
    M_p_c_t : torch.Tensor
        Low-rank predictive covariance factors of shape
        ``[batch, latents, rank]``.
    K : torch.Tensor
        Canonical precision factors shaped ``[batch, latents, rank]``.

    Returns
    -------
    torch.Tensor
        Smoothed covariance matrix of shape ``[batch, latents, latents]``.
    """
    # TODO: optimize order of operations
    P_p_t = M_p_c_t @ M_p_c_t.mT + torch.diag(Q_diag)
    I_pl_triple = torch.eye(K.shape[-1]) + K.mT @ P_p_t @ K
    Psi_t = linalg_utils.triangular_inverse(torch.linalg.cholesky(I_pl_triple)).mT
    P_s_t = P_p_t - P_p_t @ K @ Psi_t @ Psi_t.mT @ K.mT @ P_p_t

    return P_s_t


def get_P_s_1(Q_0_diag, K):
    """
    Compute the smoothed covariance for the initial time step.

    Parameters
    ----------
    Q_0_diag : torch.Tensor
        Diagonal initial covariance entries shaped ``[latents]``.
    K : torch.Tensor
        Canonical precision factors shaped ``[batch, latents, rank]``.

    Returns
    -------
    torch.Tensor
        Initial smoothed covariance matrix with shape
        ``[batch, latents, latents]``.
    """
    # TODO: optimize order of operations
    P_p_t = torch.diag(Q_0_diag)
    I_pl_triple = torch.eye(K.shape[-1]) + K.mT @ P_p_t @ K
    Psi_t = linalg_utils.triangular_inverse(torch.linalg.cholesky(I_pl_triple)).mT
    P_s_t = P_p_t - P_p_t @ K @ Psi_t @ Psi_t.mT @ K.mT @ P_p_t
    return P_s_t


"""big L"""


def fast_J_p_bqp(M_p_c, Q_inv_diag, Psi_p, v):
    """
    Evaluate the quadratic form ``(m_f - m_p)^T J_p (m_f - m_p)``.

    Parameters
    ----------
    M_p_c : torch.Tensor
        Predictive covariance factors shaped ``[batch, latents, rank]``.
    Q_inv_diag : torch.Tensor
        Inverse process noise diagonal entries.
    Psi_p : torch.Tensor
        Inverse Cholesky factor of the predictive precision matrix.
    v : torch.Tensor
        Difference vector ``m_f - m_p`` with shape ``[batch, latents]``.

    Returns
    -------
    torch.Tensor
        Quadratic form values per batch element.
    """
    qp_1 = bip(Q_inv_diag[None, :] * v, v)

    Q_inv_M_p = Q_inv_diag[None, :, None] * M_p_c
    u = bmv(Psi_p.mT @ Q_inv_M_p.mT, v)
    qp_2 = bip(u, u)

    qp = qp_1 - qp_2
    return qp


# @torch.jit.script
def fast_tr_J_p_P_f(M_p_c, K, Psi_f, Q_sqrt_diag):
    """
    Compute ``tr(J_p P_f)`` for the low-rank KL divergence term.

    Parameters
    ----------
    M_p_c : torch.Tensor
        Predictive covariance factors shaped ``[batch, latents, rank]``.
    K : torch.Tensor
        Canonical statistics with shape ``[batch, latents, rank]``.
    Psi_f : torch.Tensor
        Inverse Cholesky factor of the filtered precision matrix of shape
        ``[batch, rank, rank]``.
    Q_sqrt_diag : torch.Tensor
        Square-root process noise diagonal entries with shape ``[latents]``.

    Returns
    -------
    torch.Tensor
        Trace contribution per batch element.
    """
    L = Q_sqrt_diag.shape[-1]

    K_Psi = K @ Psi_f
    triple_1 = K_Psi.mT @ M_p_c
    tr_1 = torch.sum(triple_1.pow(2), dim=[-1, -2])

    triple_2 = Q_sqrt_diag[None, :, None] * K_Psi
    tr_2 = torch.sum(triple_2.pow(2), dim=[-1, -2])

    tr = L - tr_1 - tr_2
    return tr


# @torch.jit.script
def low_rank_kl_step_t(m_f, m_p, M_p_c, K, Psi_f, Psi_p, Q_inv_diag, Q_sqrt_diag):
    """
    Compute the KL divergence between filtered and predictive states at time ``t``.

    Parameters
    ----------
    m_f : torch.Tensor
        Filtered posterior means shaped ``[batch, latents]``.
    m_p : torch.Tensor
        Predictive means shaped ``[batch, latents]``.
    M_p_c : torch.Tensor
        Predictive covariance factors shaped ``[batch, latents, rank]``.
    K : torch.Tensor
        Canonical statistics of shape ``[batch, latents, rank]``.
    Psi_f : torch.Tensor
        Inverse Cholesky factor of the filtered precision matrix.
    Psi_p : torch.Tensor
        Inverse Cholesky factor of the predictive precision matrix.
    Q_inv_diag : torch.Tensor
        Inverse process noise diagonal entries.
    Q_sqrt_diag : torch.Tensor
        Square root of process noise diagonal entries.

    Returns
    -------
    torch.Tensor
        KL divergence values per batch item.
    """
    L = m_f.shape[-1]
    tr = fast_tr_J_p_P_f(M_p_c, K, Psi_f, Q_sqrt_diag)
    qp = fast_J_p_bqp(M_p_c, Q_inv_diag, Psi_p, m_f - m_p)
    logdet = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_f, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )
    kl = 0.5 * (qp + tr + logdet - L)
    return kl


# @torch.jit.script
def low_rank_kl_step_0(m_f, m_p, P_p_diag, K, Psi_f):
    """
    Compute the KL divergence at the initial time step using diagonal covariances.

    Parameters
    ----------
    m_f : torch.Tensor
        Filtered posterior means shaped ``[batch, latents]``.
    m_p : torch.Tensor
        Predictive means shaped ``[batch, latents]``.
    P_p_diag : torch.Tensor
        Predictive covariance diagonal entries.
    K : torch.Tensor
        Canonical statistics of shape ``[batch, latents, rank]``.
    Psi_f : torch.Tensor
        Inverse Cholesky factor of the filtered precision matrix.

    Returns
    -------
    torch.Tensor
        KL divergence values per batch item.
    """
    #  tr(J_p @ P_f) = L - tr(P_p_sqrt.mT @ K @ Psi_f @ Psi_f.mT @ K.mT @ P_p_sqrt)
    L = m_f.shape[-1]
    delta_m = m_f - m_p
    P_p_inv_diag = 1 / P_p_diag
    P_p_sqrt_diag = torch.sqrt(P_p_diag)

    K_Psi = K @ Psi_f
    triple = P_p_sqrt_diag[None, :, None] * K_Psi
    tr = L - torch.sum(triple.pow(2), dim=[-1, -2])
    qp = bip(P_p_inv_diag[None, :] * delta_m, delta_m)
    logdet = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_f, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )

    kl = 0.5 * (qp + tr + logdet - L)
    return kl


# @torch.jit.script
def fast_bmv_P_p(M_c_p, Q_diag, v):
    """
    Multiply a vector by the predictive covariance ``P_p``.

    Parameters
    ----------
    M_c_p : torch.Tensor
        Covariance factor matrix shaped ``[batch, latents, rank]``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    v : torch.Tensor
        Vector to multiply with shape matching the latents dimension.

    Returns
    -------
    torch.Tensor
        Product ``P_p v`` with the same shape as ``v``.
    """
    u_1 = bmv(M_c_p, bmv(M_c_p.mT, v))
    u_2 = Q_diag * v
    u = u_1 + u_2
    return u


# @torch.jit.script
def fast_bmv_P_f(K, Psi_f, M_c_p, Q_diag, v):
    """
    Multiply a vector by the filtered covariance ``P_f``.

    Parameters
    ----------
    K : torch.Tensor
        Canonical statistics shaped ``[batch, latents, rank]``.
    Psi_f : torch.Tensor
        Inverse Cholesky factor of the filtered precision matrix.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    v : torch.Tensor
        Vector to multiply with shape matching the latents dimension.

    Returns
    -------
    torch.Tensor
        Product ``P_f v`` with the same shape as ``v``.
    """
    u_1 = fast_bmv_P_p(M_c_p, Q_diag, v)

    triple_bmv = bmv(K, bmv(Psi_f, bmv(Psi_f.mT, bmv(K.mT, u_1))))
    u_2 = fast_bmv_P_p(M_c_p, Q_diag, triple_bmv)
    u = u_1 - u_2
    return u


# @torch.jit.script
def fast_bmv_P_p_inv(Q_diag, M_c_p, Psi_p, v):
    """
    Multiply a vector by the predictive precision ``J_p``.

    Parameters
    ----------
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Psi_p : torch.Tensor
        Inverse Cholesky factor of the predictive precision matrix.
    v : torch.Tensor
        Vector to multiply with shape matching the latents dimension.

    Returns
    -------
    torch.Tensor
        Product ``J_p v`` with the same shape as ``v``.
    """
    Q_inv_diag = 1 / Q_diag

    u_1 = Q_inv_diag * v
    u_2 = Q_inv_diag * bmv(M_c_p, bmv(Psi_p, bmv(Psi_p.mT, bmv(M_c_p.mT, u_1))))
    u = u_1 - u_2
    return u


# @torch.jit.script
def fast_P_f_diagonal(K, Psi_f, M_c_p, Q_diag):
    """
    Extract the diagonal of the filtered covariance ``P_f`` efficiently.

    Parameters
    ----------
    K : torch.Tensor
        Canonical statistics shaped ``[batch, latents, rank]``.
    Psi_f : torch.Tensor
        Inverse Cholesky factor of the filtered precision matrix.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    torch.Tensor
        Diagonal entries of ``P_f`` with shape ``[batch, latents]``.
    """
    L = K.shape[-2]
    e_basis = torch.eye(L).view(L, L)
    p = torch.stack(
        [fast_bmv_P_f(K, Psi_f, M_c_p, Q_diag, e_basis[i])[..., i] for i in range(L)],
        dim=-1,
    )
    return p


# @torch.jit.script
def fast_bmv_P_f_0(K, Psi_f, P_p_diag, v):
    """
    Multiply a vector by the filtered covariance at time zero.

    Parameters
    ----------
    K : torch.Tensor
        Canonical statistics shaped ``[batch, latents, rank]``.
    Psi_f : torch.Tensor
        Inverse Cholesky factor of the filtered precision matrix.
    P_p_diag : torch.Tensor
        Predictive covariance diagonal entries.
    v : torch.Tensor
        Vector to multiply, matching the latents dimension.

    Returns
    -------
    torch.Tensor
        Product ``P_f v`` evaluated at the initial time step.
    """
    u_1 = P_p_diag * v

    triple_bmv = bmv(K, bmv(Psi_f, bmv(Psi_f.mT, bmv(K.mT, u_1))))
    u_2 = P_p_diag * triple_bmv
    u = u_1 - u_2
    return u


# @torch.jit.script
def fast_P_f_0_diagonal(K, Psi_f, P_p_diag):
    """
    Extract the diagonal of the filtered covariance at time zero.

    Parameters
    ----------
    K : torch.Tensor
        Canonical statistics shaped ``[batch, latents, rank]``.
    Psi_f : torch.Tensor
        Inverse Cholesky factor of the filtered precision matrix.
    P_p_diag : torch.Tensor
        Predictive covariance diagonal entries.

    Returns
    -------
    torch.Tensor
        Diagonal entries of ``P_f`` with shape ``[batch, latents]``.
    """
    L = K.shape[-2]
    e_basis = torch.eye(L).view(L, L)
    p = torch.stack(
        [fast_bmv_P_f_0(K, Psi_f, P_p_diag, e_basis[i])[..., i] for i in range(L)],
        dim=-1,
    )
    return p


# @torch.jit.script
def fast_update_step(z_p_c, h_p, k, K, w_f, M_c_p, Q_diag):
    """
    Perform the low-rank measurement update.

    Parameters
    ----------
    z_p_c : torch.Tensor
        Predictive samples centered at the predictive mean with shape
        ``[n_samples, batch, latents]``.
    h_p : torch.Tensor
        Predictive natural parameters shaped ``[batch, latents]``.
    k : torch.Tensor
        Observation natural parameters shaped ``[batch, latents]``.
    K : torch.Tensor
        Observation precision factors with shape ``[batch, latents, rank]``.
    w_f : torch.Tensor
        Standard normal noise used to sample filtered latents, with shape
        ``[n_samples, batch, rank]``.
    M_c_p : torch.Tensor
        Predictive covariance factors shaped ``[batch, latents, rank]``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Posterior mean ``m``, sampled latent ``z``, and inverse Cholesky factor
        ``Psi`` of the filtered precision matrix.
    """
    n_trials, n_latents, rank = K.shape
    Q_diag_sqrt = torch.sqrt(Q_diag)
    I_r = torch.eye(rank)

    h = h_p + k

    K_mT_M_c = K.mT @ M_c_p
    K_mT_Q_sqrt = K.mT * Q_diag_sqrt[None, None, :]
    K_mT_P_K = K_mT_M_c @ K_mT_M_c.mT + K_mT_Q_sqrt @ K_mT_Q_sqrt.mT
    I_r_pl_triple = I_r + K_mT_P_K
    I_r_pl_triple_chol, _ = torch.linalg.cholesky_ex(I_r_pl_triple)
    Psi = linalg_utils.triangular_inverse(I_r_pl_triple_chol).mT

    m = fast_bmv_P_f(K, Psi, M_c_p, Q_diag, h)

    v_1 = bmv(K.mT, z_p_c) + w_f
    # z = m + z_p_c - bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    unscaled_update = bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    scaled_update = fast_bmv_P_p(M_c_p, Q_diag, unscaled_update)
    z = m + z_p_c - scaled_update
    return m, z, Psi


# @torch.jit.script
def fast_predict_step(m_theta_z_tm1, w_p_1, w_p_2, Q_diag):
    """
    Propagate the low-rank predictive distribution forward one step.

    Parameters
    ----------
    m_theta_z_tm1 : torch.Tensor
        Dynamics output samples of shape ``[n_samples, batch, latents]``.
    w_p_1 : torch.Tensor
        Standard normal samples for low-rank components,
        shape ``[n_samples, batch, n_samples]``.
    w_p_2 : torch.Tensor
        Standard normal samples for diagonal components,
        shape ``[n_samples, batch, latents]``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Centered predictive samples ``z_p_c``, predictive mean ``m_p``,
        predictive natural parameters ``h_p``, covariance factors ``M_c``,
        and inverse Cholesky factor ``Psi_p``.
    """
    S = w_p_1.shape[-1]
    sqrt_S_inv = math.sqrt(1 / S)
    Q_diag_sqrt = torch.sqrt(Q_diag)
    I_S = torch.eye(S)

    m_p = m_theta_z_tm1.mean(dim=-1)
    M_c = sqrt_S_inv * (m_theta_z_tm1 - m_p.unsqueeze(-1))

    M_c_mT_Q_inv = M_c.mT * (1 / Q_diag)
    # I_pl_MmTQinvM_chol = torch.linalg.cholesky(I_S + M_c_mT_Q_inv @ M_c)
    I_pl_MmTQinvM_chol, _ = torch.linalg.cholesky_ex(I_S + M_c_mT_Q_inv @ M_c)
    Psi_p = linalg_utils.triangular_inverse(I_pl_MmTQinvM_chol).mT
    h_p = fast_bmv_P_p_inv(Q_diag, M_c, Psi_p, m_p)

    z_p_c = bmv(M_c, w_p_1) + Q_diag_sqrt * w_p_2

    return z_p_c, m_p, h_p, M_c, Psi_p


# @torch.jit.script
def fast_filter_step_t(m_theta_z_tm1, k, K, Q_diag, t_mask):
    """
    Execute a single low-rank filtering step for ``t > 0``.

    Parameters
    ----------
    m_theta_z_tm1 : torch.Tensor
        Dynamics samples from the previous step,
        shape ``[n_samples, batch, latents]``.
    k : torch.Tensor
        Observation natural parameters for time ``t``.
    K : torch.Tensor
        Observation precision factors for time ``t``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    t_mask : bool
        When True, skip the measurement update (missing data).

    Returns
    -------
    tuple[torch.Tensor, ...]
        Sampled latent ``z_f``, posterior mean ``m_f``, predictive mean ``m_p``,
        covariance factors ``M_c_p``, filtered inverse Cholesky ``Psi_f``,
        and predictive inverse Cholesky ``Psi_p``.
    """
    n_trials, n_latents, rank = K.shape
    n_samples = m_theta_z_tm1.shape[-1]
    batch_sz = [n_trials]

    w_f = torch.randn([n_samples] + batch_sz + [rank])
    w_p_1 = torch.randn([n_samples] + batch_sz + [n_samples])
    w_p_2 = torch.randn([n_samples] + batch_sz + [n_latents])

    z_p_c, m_p, h_p, M_c_p, Psi_p = fast_predict_step(
        m_theta_z_tm1, w_p_1, w_p_2, Q_diag
    )

    if not t_mask:
        m_f, z_f, Psi_f = fast_update_step(z_p_c, h_p, k, K, w_f, M_c_p, Q_diag)
    else:
        m_f = m_p
        z_f = m_p + z_p_c
        Psi_f = torch.ones((n_trials, rank, rank)) * torch.eye(rank)

    return z_f, m_f, m_p, M_c_p, Psi_f, Psi_p


# @torch.jit.script
def fast_update_step_0(z_p_c, h_p, k, K, w_f, P_p_diag):
    """
    Perform the measurement update for the initial time step in low-rank form.

    Parameters
    ----------
    z_p_c : torch.Tensor
        Predictive centered samples for time zero.
    h_p : torch.Tensor
        Predictive natural parameters at time zero.
    k : torch.Tensor
        Observation natural parameters at time zero.
    K : torch.Tensor
        Observation precision factors at time zero.
    w_f : torch.Tensor
        Standard normal samples used to draw filtered latents.
    P_p_diag : torch.Tensor
        Predictive covariance diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Posterior mean ``m``, sampled latent ``z``, and inverse Cholesky factor
        ``Psi`` of the filtered precision matrix.
    """
    n_trials, n_latents, rank = K.shape
    I_r = torch.eye(rank)

    h = h_p + k
    P_p_K = P_p_diag[None, :, None] * K

    K_mT_P_K = K.mT @ P_p_K
    I_r_pl_triple = I_r + K_mT_P_K
    I_r_pl_triple_chol = torch.linalg.cholesky(I_r_pl_triple)
    Psi = linalg_utils.triangular_inverse(I_r_pl_triple_chol).mT

    m_1 = P_p_diag * h
    m_2 = bmv(P_p_K, chol_bmv_solve(I_r_pl_triple_chol, bmv(P_p_K.mT, h)))
    m = m_1 - m_2

    v_1 = bmv(K.mT, z_p_c) + w_f
    # z = m + z_p_c - bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    unscaled_update = bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    scaled_update = P_p_diag * unscaled_update
    z = m + z_p_c - scaled_update

    return m, z, Psi


# @torch.jit.script
def fast_filter_step_0(
    m_0: torch.Tensor,
    k: torch.Tensor,
    K: torch.Tensor,
    P_p_diag: torch.Tensor,
    n_samples: int,
):
    """
    Execute the initial low-rank filtering step.

    Parameters
    ----------
    m_0 : torch.Tensor
        Initial mean with shape ``[latents]``.
    k : torch.Tensor
        Observation natural parameters at time zero.
    K : torch.Tensor
        Observation precision factors at time zero.
    P_p_diag : torch.Tensor
        Predictive covariance diagonal entries.
    n_samples : int
        Number of latent samples to draw.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Sampled latent ``z_f``, posterior mean ``m_f``, predictive mean ``m_p``,
        filtered inverse Cholesky ``Psi_f``, and predictive covariance diagonal.
    """
    n_trials, n_latents, rank = K.shape
    batch_sz = [n_trials]
    w_p = torch.randn([n_samples] + batch_sz + [n_latents])

    z_p_c = torch.sqrt(P_p_diag) * w_p
    J_p_diag = 1 / P_p_diag
    m_p = m_0 * torch.ones(batch_sz + [n_latents])
    h_p = J_p_diag * m_p

    w_f = torch.randn([n_samples] + batch_sz + [rank])
    m_f, z_f, Psi_f = fast_update_step_0(z_p_c, h_p, k, K, w_f, P_p_diag)

    return z_f, m_f, m_p, Psi_f, P_p_diag


"""small L"""


def full_rank_mvn_kl(m_f, P_f_chol, m_p, P_p_chol):
    """
    Compute the KL divergence between two full-rank Gaussian posteriors.

    Parameters
    ----------
    m_f : torch.Tensor
        Posterior means after incorporating observations, shape
        ``[batch, time, latents]``.
    P_f_chol : torch.Tensor
        Cholesky factor of posterior covariances with the same leading shape.
    m_p : torch.Tensor
        Predictive means prior to the observation update.
    P_p_chol : torch.Tensor
        Cholesky factor of predictive covariances.

    Returns
    -------
    torch.Tensor
        KL divergence per batch and time step.
    """
    tr = torch.einsum(
        "...ii -> ...", torch.cholesky_solve(P_f_chol @ P_f_chol.mT, P_p_chol)
    )
    logdet1 = 2 * torch.sum(
        torch.log(torch.diagonal(P_f_chol, dim1=-2, dim2=-1)), dim=-1
    )
    logdet2 = 2 * torch.sum(
        torch.log(torch.diagonal(P_p_chol, dim1=-2, dim2=-1)), dim=-1
    )
    qp = bip(m_f - m_p, chol_bmv_solve(P_p_chol, m_f - m_p))
    kl = 0.5 * (tr + qp + logdet2 - logdet1 - m_f.shape[-1])

    return kl


def predict_step_t(m_theta_z_tm1, Q_diag):
    """
    Propagate the Gaussian predictive distribution one step forward.

    Parameters
    ----------
    m_theta_z_tm1 : torch.Tensor
        Samples from the dynamics output with shape ``[n_samples, latents]``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Predictive mean ``m_p`` and covariance matrix ``P_p``.
    """
    M = -0.5 * (torch.diag(Q_diag) + bop(m_theta_z_tm1, m_theta_z_tm1))

    m_p = m_theta_z_tm1.mean(dim=0)
    M_p = M.mean(dim=0)
    P_p = -2 * M_p - bop(m_p, m_p)
    return m_p, P_p


def filter_step_t(m_theta_z_tm1, k, K, Q_diag, t_mask):
    """
    Run a full-rank filtering update for time ``t > 0``.

    Parameters
    ----------
    m_theta_z_tm1 : torch.Tensor
        Dynamics samples of shape ``[n_samples, batch, latents]``.
    k : torch.Tensor
        Observation natural parameters at time ``t``.
    K : torch.Tensor
        Observation precision factors at time ``t``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    t_mask : bool
        When True, skip the measurement update (missing observation).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Sampled latent ``z_f``, predictive mean ``m_p``, posterior mean ``m_f``,
        posterior Cholesky ``P_f_chol``, and predictive Cholesky ``P_p_chol``.
    """
    n_trials, n_latents, rank = K.shape
    n_samples = m_theta_z_tm1.shape[-1]
    batch_sz = [n_trials]

    w_f = torch.randn([n_samples] + batch_sz + [n_latents], device=m_theta_z_tm1.device)
    m_p, P_p = predict_step_t(m_theta_z_tm1.movedim(-1, 0), Q_diag)
    P_p_chol = torch.linalg.cholesky(P_p)

    if not t_mask:
        h_p = chol_bmv_solve(P_p_chol, m_p)
        h_f = h_p + k

        J_p = torch.cholesky_inverse(P_p_chol)
        J_f = J_p + K @ K.mT
        J_f_chol = torch.linalg.cholesky(J_f)
        P_f_chol = linalg_utils.triangular_inverse(J_f_chol).mT
        m_f = chol_bmv_solve(J_f_chol, h_f)
    else:
        m_f = m_p
        P_f_chol = P_p_chol

    z_f = m_f + bmv(P_f_chol, w_f)

    return z_f, m_p, m_f, P_f_chol, P_p_chol


# @torch.jit.script
def filter_step_0(
    m_0: torch.Tensor,
    k: torch.Tensor,
    K: torch.Tensor,
    P_0_diag: torch.Tensor,
    n_samples: int,
):
    """
    Perform the initial full-rank filtering step.

    Parameters
    ----------
    m_0 : torch.Tensor
        Initial mean vector.
    k : torch.Tensor
        Observation natural parameters at time zero.
    K : torch.Tensor
        Observation precision factors at time zero.
    P_0_diag : torch.Tensor
        Diagonal covariance entries for the initial state.
    n_samples : int
        Number of latent samples to draw.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Sampled latent ``z_f``, posterior mean ``m_f``, posterior Cholesky
        ``P_f_chol``, and predictive Cholesky ``P_p_chol``.
    """
    n_trials, n_latents, rank = K.shape
    batch_sz = [n_trials]

    J_0_diag = 1 / P_0_diag
    h_0 = J_0_diag * m_0
    J_f = torch.diag(J_0_diag) + K @ K.mT
    J_f_chol = torch.linalg.cholesky(J_f)
    P_f_chol = linalg_utils.triangular_inverse(J_f_chol).mT

    h_f = h_0 + k
    m_f = chol_bmv_solve(J_f_chol, h_f)

    P_p_chol = torch.diag(torch.sqrt(P_0_diag)) + torch.zeros_like(P_f_chol)
    w_f = torch.randn([n_samples] + batch_sz + [n_latents], device=m_f.device)
    z_f = m_f + bmv(P_f_chol, w_f)

    return z_f, m_f, P_f_chol, P_p_chol
