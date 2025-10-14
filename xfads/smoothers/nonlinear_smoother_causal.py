"""
Causal smoother variants for low-rank nonlinear state-space models.

The routines in this module implement causal filtering and smoothing updates
that operate on canonical statistics produced by encoder networks. They are
used for sequential inference where future observations are not incorporated.
"""

import math
import torch
from torch import nn
import torch.nn.functional as Fn
from .. import linalg_utils
from ..linalg_utils import bmv, bip, chol_bmv_solve


class LowRankNonlinearStateSpaceModel(nn.Module):
    """
    Causal variational smoother for low-rank nonlinear state-space models.

    This module only incorporates past observations when inferring latent
    trajectories, enabling streaming or online inference scenarios.

    Parameters
    ----------
    dynamics_mod : nn.Module
        Latent dynamics module with ``mean_fn`` and ``log_Q`` parameters.
    likelihood_pdf : nn.Module
        Emission model supporting ``get_ell`` for log-likelihood evaluation.
    initial_c_pdf : nn.Module
        Initial latent distribution exposing ``m_0`` and ``log_Q_0`` tensors.
    backward_encoder : nn.Module
        Encoder producing backward canonical statistics ``(k_b, K_b)``.
    local_encoder : nn.Module
        Encoder producing forward canonical statistics ``(k_y, K_y)``.
    nl_filter : nn.Module
        Nonlinear filter operating on canonical statistics.
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
        Initialize the causal smoother components.

        Parameters
        ----------
        dynamics_mod : nn.Module
            Latent dynamics module.
        likelihood_pdf : nn.Module
            Emission likelihood module.
        initial_c_pdf : nn.Module
            Initial latent distribution.
        backward_encoder : nn.Module
            Network providing backward canonical statistics.
        local_encoder : nn.Module
            Network providing forward canonical statistics.
        nl_filter : nn.Module
            Filtering backend for canonical statistics.
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
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        p_mask_apb: float = 0.0,
        get_P_s: bool = False,
    ):
        """
        Compute the ELBO and smoothed latents given observations.

        Parameters
        ----------
        y : torch.Tensor
            Observation tensor with shape ``[batch, time, neurons]``.
        n_samples : int
            Number of latent samples drawn by the filter.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        p_mask_apb : float, optional
            Dropout probability for combined statistics (unused), by default 0.0.
        get_P_s : bool, optional
            If True, request smoothed covariance estimates.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Loss scalar, smoothed latent samples, and statistics dictionary.
        """
        z_s, stats = self.fast_smooth_1_to_T(
            y,
            n_samples,
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

    @torch.jit.export
    def forward_filter(
        self,
        y,
        n_samples: int,
        p_mask_y_in: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        p_mask_apb: float = 0.0,
    ):
        """
        Run causal filtering without smoothing the future.

        Parameters
        ----------
        y : torch.Tensor
            Observation tensor with shape ``[batch, time, neurons]``.
        n_samples : int
            Number of latent samples drawn by the filter.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        p_mask_apb : float, optional
            Unused dropout probability for API compatibility, by default 0.0.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Loss scalar, filtered latent samples, and statistics dictionary.
        """
        z_f, stats = self.fast_filter_1_to_T(
            y,
            n_samples,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_b=p_mask_b,
            get_kl=True,
        )

        ell = self.likelihood_pdf.get_ell(y, z_f).mean(dim=0)
        loss = stats["kl"] - ell
        loss = loss.sum(dim=-1).mean()
        return loss, z_f, stats

    @torch.jit.export
    def fast_smooth_1_to_T(
        self,
        y,
        n_samples: int,
        p_mask_a: float = 0.0,
        p_mask_y_in: float = 0.0,
        p_mask_b: float = 0.0,
        get_kl: bool = False,
        get_v: bool = False,
        get_P_s: bool = False,
    ):
        """
        Run the causal smoother across all time steps.

        Parameters
        ----------
        y : torch.Tensor
            Observation tensor shaped ``[batch, time, neurons]``.
        n_samples : int
            Number of samples produced by the nonlinear filter.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        get_kl : bool, optional
            If True, include KL divergence statistics, by default False.
        get_v : bool, optional
            Unused placeholder for variance requests, by default False.
        get_P_s : bool, optional
            If True, include smoothed covariance estimates in the stats dict.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Smoothed latent samples and associated statistics.
        """
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=y.device))
        t_mask_b = torch.bernoulli((1 - p_mask_b) * torch.ones((n_trials, n_time_bins), device=y.device))
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons), device=y.device)
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b
        K_b = t_mask_b[..., None, None] * K_b

        z_s, stats = self.nl_filter(
            k_y, K_y, k_b, K_b, n_samples, get_kl=get_kl, get_P_s=get_P_s
        )
        return z_s, stats

    @torch.jit.export
    def fast_filter_1_to_T(
        self,
        y,
        n_samples: int,
        p_mask_a: float = 0.0,
        p_mask_y_in: float = 0.0,
        p_mask_b: float = 0.0,
        get_kl: bool = False,
        get_v: bool = False,
    ):
        """
        Run the causal filter to produce latent trajectories.

        Parameters
        ----------
        y : torch.Tensor
            Observation tensor shaped ``[batch, time, neurons]``.
        n_samples : int
            Number of latent samples produced by the filter.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        get_kl : bool, optional
            If True, include KL divergence terms, by default False.
        get_v : bool, optional
            Unused placeholder for predictive variance, by default False.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Filtered latent samples and statistics dictionary.
        """
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=y.device))
        t_mask_b = torch.bernoulli((1 - p_mask_b) * torch.ones((n_trials, n_time_bins), device=y.device))
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons), device=y.device)
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)

        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b * 0.0
        K_b = t_mask_b[..., None, None] * K_b * 0.0

        z_s, stats = self.nl_filter(k_y, K_y, k_b, K_b, n_samples, get_kl=get_kl)
        return z_s, stats

    def predict_forward(self, z_tm1: torch.Tensor, n_bins: int):
        """
        Simulate latent trajectories forward using the dynamics model.

        Parameters
        ----------
        z_tm1 : torch.Tensor
            Initial latent state with shape ``[batch, latents]``.
        n_bins : int
            Number of future steps to generate.

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
                    z_tm1, device=z_tm1.device
                )
            else:
                z_t = self.dynamics_mod.mean_fn(
                    z_forward[t - 1]
                ) + Q_sqrt * torch.randn_like(z_forward[t - 1], device=z_tm1.device)

            z_forward.append(z_t)

        z_forward = torch.stack(z_forward, dim=2)
        return z_forward


class LowRankNonlinearStateSpaceModelWithInput(LowRankNonlinearStateSpaceModel):
    """
    Causal smoother variant that conditions on exogenous input signals.

    Inputs are incorporated through the nonlinear filter, enabling behaviour
    models that depend on external controls or stimuli.
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
        Parameters
        ----------
        dynamics_mod : nn.Module
            Latent dynamics module.
        likelihood_pdf : nn.Module
            Emission likelihood module.
        initial_c_pdf : nn.Module
            Initial latent distribution.
        backward_encoder : nn.Module
            Backward encoder providing canonical statistics.
        local_encoder : nn.Module
            Forward encoder providing canonical statistics.
        nl_filter : nn.Module
            Input-aware nonlinear filtering backend.
        """
        super().__init__(
            dynamics_mod,
            likelihood_pdf,
            initial_c_pdf,
            backward_encoder,
            local_encoder,
            nl_filter,
        )

    @torch.jit.export
    def forward(
        self,
        y,
        u,
        n_samples: int,
        p_mask_y_in: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        p_mask_apb: float = 0.0,
    ):
        """
        Compute the ELBO conditioned on observations and inputs.

        Parameters
        ----------
        y : torch.Tensor
            Observation tensor shaped ``[batch, time, neurons]``.
        u : torch.Tensor
            Input tensor aligned with the observation time axis.
        n_samples : int
            Number of latent samples drawn by the filter.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        p_mask_apb : float, optional
            Unused dropout probability for API compatibility, by default 0.0.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Loss scalar, smoothed latent samples, and statistics dictionary.
        """
        z_s, stats = self.fast_smooth_1_to_T(
            y,
            u,
            n_samples,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_b=p_mask_b,
            get_kl=True,
        )
        ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)
        loss = stats["kl"] - ell
        loss = loss.sum(dim=-1).mean()
        return loss, z_s, stats

    @torch.jit.export
    def forward_filter(
        self,
        y,
        u,
        n_samples: int,
        p_mask_y_in: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        p_mask_apb: float = 0.0,
    ):
        """
        Run causal filtering with inputs and compute the ELBO contribution.

        Parameters
        ----------
        y : torch.Tensor
            Observation tensor shaped ``[batch, time, neurons]``.
        u : torch.Tensor
            Input tensor aligned with observations.
        n_samples : int
            Number of latent samples to draw.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        p_mask_apb : float, optional
            Unused dropout probability for API compatibility, by default 0.0.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Loss scalar, filtered latent samples, and statistics dict.
        """
        z_f, stats = self.fast_filter_1_to_T(
            y,
            u,
            n_samples,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_b=p_mask_b,
            get_kl=True,
        )

        ell = self.likelihood_pdf.get_ell(y, z_f).mean(dim=0)
        loss = stats["kl"] - ell
        loss = loss.sum(dim=-1).mean()
        return loss, z_f, stats

    @torch.jit.export
    def fast_smooth_1_to_T(
        self,
        y,
        u,
        n_samples: int,
        p_mask_a: float = 0.0,
        p_mask_y_in: float = 0.0,
        p_mask_b: float = 0.0,
        get_kl: bool = False,
        get_v: bool = False,
    ):
        """
        Run the causal smoother conditioned on inputs.

        Parameters
        ----------
        y : torch.Tensor
            Observation tensor shaped ``[batch, time, neurons]``.
        u : torch.Tensor
            Input tensor aligned with ``y``.
        n_samples : int
            Number of latent samples produced by the filter.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        get_kl : bool, optional
            If True, include KL divergence terms, by default False.
        get_v : bool, optional
            Unused placeholder for predictive variance, by default False.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Smoothed latent samples and statistics dictionary.
        """
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=y.device))
        t_mask_b = torch.bernoulli((1 - p_mask_b) * torch.ones((n_trials, n_time_bins), device=y.device))
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons), device=y.device)
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b
        K_b = t_mask_b[..., None, None] * K_b

        z_s, stats = self.nl_filter(u, k_y, K_y, k_b, K_b, n_samples, get_kl=get_kl)
        return z_s, stats

    @torch.jit.export
    def fast_filter_1_to_T(
        self,
        y,
        u,
        n_samples: int,
        p_mask_a: float = 0.0,
        p_mask_y_in: float = 0.0,
        p_mask_b: float = 0.0,
        get_kl: bool = False,
        get_v: bool = False,
    ):
        """
        Run the causal filter conditioned on inputs.

        Parameters
        ----------
        y : torch.Tensor
            Observation tensor shaped ``[batch, time, neurons]``.
        u : torch.Tensor
            Input tensor aligned with the time dimension.
        n_samples : int
            Number of latent samples drawn by the filter.
        p_mask_a : float, optional
            Dropout probability for forward encoder statistics, by default 0.0.
        p_mask_y_in : float, optional
            Observation dropout probability, by default 0.0.
        p_mask_b : float, optional
            Dropout probability for backward encoder statistics, by default 0.0.
        get_kl : bool, optional
            If True, return KL divergence terms, by default False.
        get_v : bool, optional
            Unused placeholder for predictive variance, by default False.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Filtered latent samples and statistics dictionary.
        """
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=y.device))
        t_mask_b = torch.bernoulli((1 - p_mask_b) * torch.ones((n_trials, n_time_bins), device=y.device))
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons), device=y.device)
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)

        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b * 0.0
        K_b = t_mask_b[..., None, None] * K_b * 0.0

        z_s, stats = self.nl_filter(u, k_y, K_y, k_b, K_b, n_samples, get_kl=get_kl)
        return z_s, stats

    def predict_forward(self, z_tm1: torch.Tensor, u: torch.Tensor, n_bins: int):
        """
        Simulate latent trajectories forward while incorporating inputs.

        Parameters
        ----------
        z_tm1 : torch.Tensor
            Latent state at time ``t-1`` with shape ``[batch, latents]``.
        u : torch.Tensor
            Input sequence shaped ``[batch, time, input_dim]``.
        n_bins : int
            Number of future steps to generate.

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
                    z_tm1, device=z_tm1.device
                )
            else:
                z_t = self.dynamics_mod.mean_fn(
                    z_forward[t - 1]
                ) + Q_sqrt * torch.randn_like(z_forward[t - 1], device=z_tm1.device)

            z_t += self.nl_filter.input_latent_fn(u[:, t])
            z_forward.append(z_t)

        z_forward = torch.stack(z_forward, dim=2)
        return z_forward


def get_P_s_t(Q_diag, M_p_c_t, K_a, K_b):
    """
    Compute the smoothed covariance for time ``t`` in the causal model.

    Parameters
    ----------
    Q_diag : torch.Tensor
        Process noise diagonal entries with shape ``[latents]``.
    M_p_c_t : torch.Tensor
        Predictive covariance factors shaped ``[batch, latents, rank]``.
    K_a : torch.Tensor
        Forward canonical precision factors ``[batch, latents, rank]``.
    K_b : torch.Tensor
        Backward canonical precision factors ``[batch, latents, rank]``.

    Returns
    -------
    torch.Tensor
        Smoothed covariance matrix ``[batch, latents, latents]``.
    """
    # TODO: optimize order of operations
    K = torch.cat([K_a, K_b], dim=-1)
    P_p_t = M_p_c_t @ M_p_c_t.mT + torch.diag(Q_diag)
    I_pl_triple = torch.eye(K.shape[-1], device=K.device) + K.mT @ P_p_t @ K
    Psi_t = linalg_utils.triangular_inverse(torch.linalg.cholesky(I_pl_triple)).mT
    P_s_t = P_p_t - P_p_t @ K @ Psi_t @ Psi_t.mT @ K.mT @ P_p_t

    return P_s_t


def get_P_s_1(Q_0_diag, K_a, K_b):
    """
    Compute the smoothed covariance at the initial time step.

    Parameters
    ----------
    Q_0_diag : torch.Tensor
        Initial process noise diagonal entries.
    K_a : torch.Tensor
        Forward canonical precision factors at time zero.
    K_b : torch.Tensor
        Backward canonical precision factors at time zero.

    Returns
    -------
    torch.Tensor
        Initial smoothed covariance matrix of shape ``[batch, latents, latents]``.
    """
    # TODO: optimize order of operations
    P_p_t = torch.diag(Q_0_diag)
    K = torch.cat([K_a, K_b], dim=-1)
    I_pl_triple = torch.eye(K.shape[-1], device=K.device) + K.mT @ P_p_t @ K
    Psi_t = linalg_utils.triangular_inverse(torch.linalg.cholesky(I_pl_triple)).mT
    P_s_t = P_p_t - P_p_t @ K @ Psi_t @ Psi_t.mT @ K.mT @ P_p_t
    return P_s_t


class NonlinearFilter(nn.Module):
    """
    Causal low-rank nonlinear filter that propagates variational posteriors.

    The filter takes forward and backward canonical statistics to compute the
    smoothed latent distribution while only consuming past observations.
    """

    def __init__(self, dynamics_mod, initial_c_pdf):
        """
        Parameters
        ----------
        dynamics_mod : nn.Module
            Latent dynamics module providing ``mean_fn`` and ``log_Q``.
        initial_c_pdf : nn.Module
            Initial latent distribution exposing ``m_0`` and ``log_Q_0``.
        """
        super().__init__()

        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf

    def forward(
        self,
        k_y: torch.Tensor,
        K_y: torch.Tensor,
        k_b: torch.Tensor,
        K_b: torch.Tensor,
        n_samples: int,
        get_kl: bool = False,
        p_mask: float = 0.0,
        get_P_s: bool = False,
    ):
        """
        Perform causal filtering given encoder canonical statistics.

        Parameters
        ----------
        k_y : torch.Tensor
            Forward linear statistics with shape ``[batch, time, latents]``.
        K_y : torch.Tensor
            Forward precision factors shaped ``[batch, time, latents, rank]``.
        k_b : torch.Tensor
            Backward linear statistics with the same shape as ``k_y``.
        K_b : torch.Tensor
            Backward precision factors shaped like ``K_y``.
        n_samples : int
            Number of latent samples to draw.
        get_kl : bool, optional
            If True, compute KL divergence terms, by default False.
        p_mask : float, optional
            Unused dropout probability for API compatibility, by default 0.0.
        get_P_s : bool, optional
            If True, accumulate smoothed covariance estimates.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Smoothed latent samples and statistics including KL terms,
            posterior means, and optional covariances.
        """
        n_trials, n_time_bins, n_latents, rank_y = K_y.shape

        kl = []
        m_f = []
        m_p = []
        m_s = []
        z_f = []
        z_s = []
        Psi_f = []
        Psi_p = []
        # M_p_f_c = []

        P_s = []

        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        stats = {}

        for t in range(n_time_bins):
            if t == 0:
                m_0 = self.initial_c_pdf.m_0
                Q_0_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
                Psi_p_t = torch.zeros(
                    (n_trials, n_samples, n_samples), device=k_y.device
                )
                z_f_t, m_f_t, m_p_t, Psi_f_t, h_f_t = fast_filter_step_0(
                    m_0, k_y[:, 0], K_y[:, 0], Q_0_diag, n_samples
                )
                m_s_t, z_s_t, Psi_s_t = fast_update_filtering_to_smoothing_stats_0(
                    z_f_t,
                    h_f_t,
                    m_f_t,
                    Psi_f_t,
                    k_b[:, t],
                    K_b[:, t],
                    K_y[:, t],
                    Q_0_diag,
                )
                kl_t = low_rank_kl_step_0(
                    m_s_t, m_0, Q_0_diag, Q_diag, K_y[:, 0], K_b[:, 0], Psi_f_t, Psi_s_t
                )

                if get_P_s:
                    P_s.append(get_P_s_1(Q_0_diag, K_y[:, 0], K_b[:, 0]))
            else:
                m_fn_z_f_tm1 = self.dynamics_mod.mean_fn(z_f[t - 1]).movedim(0, -1)
                m_fn_z_s_tm1 = self.dynamics_mod.mean_fn(z_s[t - 1]).movedim(0, -1)
                z_f_t, m_f_t, m_p_t, M_p_c_t, Psi_f_t, Psi_p_t, h_f_t = (
                    fast_filter_step_t(
                        m_fn_z_f_tm1, k_y[:, t], K_y[:, t], Q_diag, torch.tensor(False)
                    )
                )
                m_s_t, z_s_t, Psi_s_t = fast_update_filtering_to_smoothing_stats_t(
                    z_f_t,
                    h_f_t,
                    m_f_t,
                    Psi_f_t,
                    M_p_c_t,
                    k_b[:, t],
                    K_b[:, t],
                    K_y[:, t],
                    Q_diag,
                )
                _, m_s_p_t, _, M_s_p_c_t, Psi_s_p_t = fast_predict_step(
                    m_fn_z_s_tm1, Q_diag
                )
                kl_t = low_rank_kl_step_t(
                    m_s_t,
                    m_s_p_t,
                    M_p_c_t,
                    M_s_p_c_t,
                    K_y[:, t],
                    K_b[:, t],
                    Psi_p_t,
                    Psi_f_t,
                    Psi_s_p_t,
                    Psi_s_t,
                    Q_diag,
                )

                if get_P_s:
                    P_s.append(get_P_s_t(Q_diag, M_p_c_t, K_y[:, t], K_b[:, t]))

            kl.append(kl_t)
            z_s.append(z_s_t)
            z_f.append(z_f_t)
            m_s.append(m_s_t)
            m_f.append(m_f_t)
            m_p.append(m_p_t)
            Psi_f.append(Psi_f_t)
            Psi_p.append(Psi_p_t)

        z_s = torch.stack(z_s, dim=2)
        stats["kl"] = torch.stack(kl, dim=1)
        stats["m_s"] = torch.stack(m_s, dim=1)
        stats["m_f"] = torch.stack(m_f, dim=1)
        stats["m_p"] = torch.stack(m_p, dim=1)
        stats["Psi_f"] = torch.stack(Psi_f, dim=1)
        stats["Psi_p"] = torch.stack(Psi_p, dim=1)

        if get_P_s:
            stats["P_s"] = torch.stack(P_s, dim=1)

        return z_s, stats


class NonlinearFilterWithInput(nn.Module):
    """
    Causal nonlinear filter that additionally conditions on external inputs.

    Inputs are mapped to latent space via ``input_latent_fn`` before combining
    with encoder statistics during filtering.
    """

    def __init__(self, input_latent_fn, dynamics_mod, initial_c_pdf):
        """
        Parameters
        ----------
        input_latent_fn : Callable
            Function mapping inputs to latent contributions.
        dynamics_mod : nn.Module
            Latent dynamics module providing ``mean_fn`` and ``log_Q``.
        initial_c_pdf : nn.Module
            Initial latent distribution exposing ``m_0`` and ``log_Q_0``.
        """
        super().__init__()

        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf
        self.input_latent_fn = input_latent_fn

    def forward(
        self,
        u: torch.Tensor,
        k_y: torch.Tensor,
        K_y: torch.Tensor,
        k_b: torch.Tensor,
        K_b: torch.Tensor,
        n_samples: int,
        get_kl: bool = False,
        p_mask: float = 0.0,
    ):
        """
        Perform causal filtering with encoder statistics and inputs.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor shaped ``[batch, time, input_dim]``.
        k_y : torch.Tensor
            Forward linear statistics ``[batch, time, latents]``.
        K_y : torch.Tensor
            Forward precision factors ``[batch, time, latents, rank]``.
        k_b : torch.Tensor
            Backward linear statistics.
        K_b : torch.Tensor
            Backward precision factors.
        n_samples : int
            Number of latent samples to draw.
        get_kl : bool, optional
            If True, compute KL divergence terms, by default False.
        p_mask : float, optional
            Unused dropout probability for compatibility, by default 0.0.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Smoothed latent samples and statistics dictionary.
        """
        n_trials, n_time_bins, n_latents, rank_y = K_y.shape

        kl = []
        m_f = []
        m_p = []
        m_s = []
        z_f = []
        z_s = []
        Psi_f = []
        Psi_p = []

        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        stats = {}

        for t in range(n_time_bins):
            input_update = self.input_latent_fn(u[:, t])

            if t == 0:
                m_0 = self.initial_c_pdf.m_0 + input_update
                Q_0_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
                Psi_p_t = torch.zeros(
                    (n_trials, n_samples, n_samples), device=k_y.device
                )
                z_f_t, m_f_t, m_p_t, Psi_f_t, h_f_t = fast_filter_step_0(
                    m_0, k_y[:, 0], K_y[:, 0], Q_0_diag, n_samples
                )
                m_s_t, z_s_t, Psi_s_t = fast_update_filtering_to_smoothing_stats_0(
                    z_f_t,
                    h_f_t,
                    m_f_t,
                    Psi_f_t,
                    k_b[:, t],
                    K_b[:, t],
                    K_y[:, t],
                    Q_0_diag,
                )
                kl_t = low_rank_kl_step_0(
                    m_s_t, m_0, Q_0_diag, Q_diag, K_y[:, 0], K_b[:, 0], Psi_f_t, Psi_s_t
                )
            else:
                m_fn_z_f_tm1 = self.dynamics_mod.mean_fn(z_f[t - 1]).movedim(
                    0, -1
                ) + input_update.unsqueeze(-1)
                m_fn_z_s_tm1 = self.dynamics_mod.mean_fn(z_s[t - 1]).movedim(
                    0, -1
                ) + input_update.unsqueeze(-1)
                z_f_t, m_f_t, m_p_t, M_p_c_t, Psi_f_t, Psi_p_t, h_f_t = (
                    fast_filter_step_t(
                        m_fn_z_f_tm1, k_y[:, t], K_y[:, t], Q_diag, torch.tensor(False)
                    )
                )
                m_s_t, z_s_t, Psi_s_t = fast_update_filtering_to_smoothing_stats_t(
                    z_f_t,
                    h_f_t,
                    m_f_t,
                    Psi_f_t,
                    M_p_c_t,
                    k_b[:, t],
                    K_b[:, t],
                    K_y[:, t],
                    Q_diag,
                )
                _, m_s_p_t, _, M_s_p_c_t, Psi_s_p_t = fast_predict_step(
                    m_fn_z_s_tm1, Q_diag
                )
                kl_t = low_rank_kl_step_t(
                    m_s_t,
                    m_s_p_t,
                    M_p_c_t,
                    M_s_p_c_t,
                    K_y[:, t],
                    K_b[:, t],
                    Psi_p_t,
                    Psi_f_t,
                    Psi_s_p_t,
                    Psi_s_t,
                    Q_diag,
                )

            kl.append(kl_t)
            z_s.append(z_s_t)
            z_f.append(z_f_t)
            m_s.append(m_s_t)
            m_f.append(m_f_t)
            m_p.append(m_p_t)
            Psi_f.append(Psi_f_t)
            Psi_p.append(Psi_p_t)

        z_s = torch.stack(z_s, dim=2)
        stats["kl"] = torch.stack(kl, dim=1)
        stats["m_s"] = torch.stack(m_s, dim=1)
        stats["m_f"] = torch.stack(m_f, dim=1)
        stats["m_p"] = torch.stack(m_p, dim=1)
        stats["Psi_f"] = torch.stack(Psi_f, dim=1)
        stats["Psi_p"] = torch.stack(Psi_p, dim=1)

        return z_s, stats


def predict_step_t(m_theta_z_tm1, Q_diag):
    """
    Propagate the smoothing distribution forward one step.

    Parameters
    ----------
    m_theta_z_tm1 : torch.Tensor
        Dynamics samples from the previous time step with shape
        ``[n_samples, batch, latents]`` or ``[n_samples, latents]``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Predictive mean ``m_p`` and covariance ``P_p`` for the next step.
    """
    S = m_theta_z_tm1.shape[-1]
    sqrt_S_inv = math.sqrt(1 / S)

    m_p = m_theta_z_tm1.mean(dim=-1)
    M_c = sqrt_S_inv * (m_theta_z_tm1 - m_p.unsqueeze(-1))
    P_p = M_c @ M_c.mT + torch.diag(Q_diag)

    return m_p, P_p


def alt_kl_step_t(m_s, m_f, m_p, a, A, B, Psi_f, Psi_s, Psi_p, M_c_p, Q_diag):
    """
    Compute the alternative KL objective at time ``t`` used for diagnostics.

    Parameters
    ----------
    m_s : torch.Tensor
        Smoothed posterior means.
    m_f : torch.Tensor
        Filtered posterior means.
    m_p : torch.Tensor
        Predictive means.
    a : torch.Tensor
        Linear term of the auxiliary distribution.
    A : torch.Tensor
        Forward canonical precision factors.
    B : torch.Tensor
        Backward canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    Psi_s : torch.Tensor
        Inverse Cholesky of the smoothed precision matrix.
    Psi_p : torch.Tensor
        Inverse Cholesky of the predictive precision matrix.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    torch.Tensor
        Alternative KL value per batch item.
    """
    P_sA = fast_bmm_P_s(Psi_f, Psi_s, B, A, M_c_p, Q_diag, A)
    P_pA = fast_bmm_P_p(M_c_p, Q_diag, A)

    AmTP_sA = A.mT @ P_sA
    AmTP_pA = A.mT @ P_pA
    AmTm_s = bmv(A.mT, m_s)
    AmTm_f = bmv(A.mT, m_f)
    P_p_inv_m_f = fast_bmv_P_p_inv(Q_diag, M_c_p, Psi_p, m_f)
    P_p_inv_m_p = fast_bmv_P_p_inv(Q_diag, M_c_p, Psi_p, m_p)
    triple_chol = torch.linalg.cholesky(
        torch.eye(A.shape[-1], device=m_s.device) + AmTP_pA
    )
    logdet_triple = -2 * torch.sum(
        torch.log(torch.diagonal(triple_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )

    inner_p = (
        bip(a, m_s)
        - 0.5 * bip(AmTm_s, AmTm_s)
        - 0.5 * torch.diagonal(AmTP_sA, dim1=-2, dim2=-1).sum(dim=-1)
    )
    delta_logZ = 0.5 * (
        bip(m_f, P_p_inv_m_f)
        - bip(m_p, P_p_inv_m_p)
        + bip(AmTm_f, AmTm_f)
        - logdet_triple
    )
    alt_kl = inner_p - delta_logZ

    return alt_kl


def alt_kl_step_0(m_s, m_f, m_0, a, A, B, Psi_f, Psi_s, Q_0_diag):
    """
    Compute the alternative KL objective for the initial time step.

    Parameters
    ----------
    m_s : torch.Tensor
        Smoothed posterior means at time zero.
    m_f : torch.Tensor
        Filtered posterior means at time zero.
    m_0 : torch.Tensor
        Prior mean of the initial state.
    a : torch.Tensor
        Linear term of the auxiliary distribution.
    A : torch.Tensor
        Forward canonical precision factors.
    B : torch.Tensor
        Backward canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    Psi_s : torch.Tensor
        Inverse Cholesky of the smoothed precision matrix.
    Q_0_diag : torch.Tensor
        Diagonal entries of the initial covariance.

    Returns
    -------
    torch.Tensor
        Alternative KL value per batch item.
    """
    P_sA = fast_bmm_P_s_0(Psi_f, Psi_s, B, A, Q_0_diag, A)
    # P_pA = A * Q_0_diag[:, None]
    P_pA = torch.diag(Q_0_diag) @ A

    AmTP_sA = A.mT @ P_sA
    AmTP_pA = A.mT @ P_pA
    AmTm_s = bmv(A.mT, m_s)
    AmTm_f = bmv(A.mT, m_f)
    P_p_inv_m_f = (1 / Q_0_diag) * m_f
    P_p_inv_m_p = (1 / Q_0_diag) * m_0
    triple_chol = torch.linalg.cholesky(
        torch.eye(A.shape[-1], device=m_s.device) + AmTP_pA
    )
    logdet_triple = -2 * torch.sum(
        torch.log(torch.diagonal(triple_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )

    inner_p = (
        bip(a, m_s)
        - 0.5 * bip(AmTm_s, AmTm_s)
        - 0.5 * torch.diagonal(AmTP_sA, dim1=-2, dim2=-1).sum(dim=-1)
    )
    delta_logZ = 0.5 * (
        bip(m_f, P_p_inv_m_f)
        - bip(m_0, P_p_inv_m_p)
        + bip(AmTm_f, AmTm_f)
        - logdet_triple
    )
    alt_kl = inner_p - delta_logZ

    return alt_kl


# @torch.jit.script
def fast_J_p_bqp(M_p_c, Q_inv_diag, Psi_p, v):
    """
    Evaluate ``v^T J_p v`` using low-rank structures.

    Parameters
    ----------
    M_p_c : torch.Tensor
        Predictive covariance factors shaped ``[batch, latents, rank]``.
    Q_inv_diag : torch.Tensor
        Inverse process noise diagonal entries.
    Psi_p : torch.Tensor
        Inverse Cholesky of the predictive precision matrix.
    v : torch.Tensor
        Vector to evaluate the quadratic form on.

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


def fast_tr_J_s_p_P_s(M_f_p_c, M_s_p_c, A, B, Psi_f, Psi_s_p, Psi_s, Q_diag):
    """
    Compute ``tr(J_s^+ P_s)`` appearing in the causal KL expression.

    Parameters
    ----------
    M_f_p_c : torch.Tensor
        Filtered covariance factors.
    M_s_p_c : torch.Tensor
        Smoothed covariance factors.
    A : torch.Tensor
        Forward canonical precision factors.
    B : torch.Tensor
        Backward canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of filtered precision.
    Psi_s_p : torch.Tensor
        Inverse Cholesky of predictive smoothed precision.
    Psi_s : torch.Tensor
        Inverse Cholesky of smoothed precision.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    torch.Tensor
        Trace values per batch element.
    """
    Q_inv_diag = 1 / Q_diag
    L = Q_inv_diag.shape[-1]
    Q_inv_sqrt_diag = torch.sqrt(Q_inv_diag)

    tr_1_sqrt = M_f_p_c * Q_inv_sqrt_diag[None, :, None]

    tr_2_sqrt = Q_inv_sqrt_diag[None, :, None] * fast_bmm_P_p(
        M_f_p_c, Q_diag, A @ Psi_f
    )
    tr_3_sqrt = Q_inv_sqrt_diag[None, :, None] * fast_bmm_P_f(
        A, Psi_f, M_f_p_c, Q_diag, B @ Psi_s
    )

    tr_4_term_1 = fast_bmm_P_s(
        Psi_f,
        Psi_s,
        B,
        A,
        M_f_p_c,
        Q_diag,
        (Q_inv_diag[None, :, None] * M_s_p_c) @ Psi_s_p,
    )
    tr_4_term_2 = Psi_s_p.mT @ (Q_inv_diag[None, :, None] * M_s_p_c).mT

    tr_1 = torch.sum(tr_1_sqrt.pow(2), dim=[-1, -2])
    tr_2 = torch.sum(tr_2_sqrt.pow(2), dim=[-1, -2])
    tr_3 = torch.sum(tr_3_sqrt.pow(2), dim=[-1, -2])
    tr_4 = torch.sum(tr_4_term_1.mT * tr_4_term_2, dim=[-1, -2])

    tr = L + tr_1 - tr_2 - tr_3 - tr_4
    return tr


def log_det_kl_t(Psi_s_p, Psi_f, Psi_s, Psi_f_p):
    """
    Compute log-determinant contributions for the causal KL term.

    Parameters
    ----------
    Psi_s_p : torch.Tensor
        Inverse Cholesky of the predictive smoothed precision.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision.
    Psi_s : torch.Tensor
        Inverse Cholesky of the smoothed precision.
    Psi_f_p : torch.Tensor
        Inverse Cholesky of the predictive filtered precision.

    Returns
    -------
    torch.Tensor
        Log-determinant contribution per batch element.
    """
    term_1 = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_s_p, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )
    term_2 = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_s, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )
    term_3 = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_f, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )
    term_4 = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_f_p, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )

    logdet = term_1 + term_2 + term_3 - term_4

    return logdet


def low_rank_kl_step_t(
    m_s, m_s_p, M_f_p_c, M_s_p_c, A, B, Psi_f_p, Psi_f, Psi_s_p, Psi_s, Q_diag
):
    """
    Compute the KL divergence at time ``t`` for the causal smoother.

    Parameters
    ----------
    m_s : torch.Tensor
        Smoothed posterior means.
    m_s_p : torch.Tensor
        Predictive smoothed means.
    M_f_p_c : torch.Tensor
        Filtered covariance factors.
    M_s_p_c : torch.Tensor
        Predictive smoothed covariance factors.
    A : torch.Tensor
        Forward canonical precision factors.
    B : torch.Tensor
        Backward canonical precision factors.
    Psi_f_p : torch.Tensor
        Inverse Cholesky of the predictive filtered precision.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision.
    Psi_s_p : torch.Tensor
        Inverse Cholesky of the predictive smoothed precision.
    Psi_s : torch.Tensor
        Inverse Cholesky of the smoothed precision.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    torch.Tensor
        KL divergence values per batch element.
    """
    Q_inv_diag = 1 / Q_diag
    L = m_s.shape[-1]

    tr = fast_tr_J_s_p_P_s(M_f_p_c, M_s_p_c, A, B, Psi_f, Psi_s_p, Psi_s, Q_diag)
    qp = fast_J_p_bqp(M_s_p_c, Q_inv_diag, Psi_s_p, m_s - m_s_p)
    log_det = log_det_kl_t(Psi_s_p, Psi_f, Psi_s, Psi_f_p)
    kl = 0.5 * (qp + tr + log_det - L)
    return kl


# @torch.jit.script
def low_rank_kl_step_0(m_s, m_0, Q_0_diag, Q_diag, A, B, Psi_f, Psi_s):
    """
    Compute the KL divergence at the initial time step for the causal smoother.

    Parameters
    ----------
    m_s : torch.Tensor
        Smoothed posterior means at time zero.
    m_0 : torch.Tensor
        Prior mean of the initial state.
    Q_0_diag : torch.Tensor
        Initial covariance diagonal entries.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    A : torch.Tensor
        Forward canonical precision factors.
    B : torch.Tensor
        Backward canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    Psi_s : torch.Tensor
        Inverse Cholesky of the smoothed precision matrix.

    Returns
    -------
    torch.Tensor
        KL divergence value per batch element.
    """
    L = m_s.shape[-1]
    delta_m = m_s - m_0
    Q_0_inv_diag = 1 / Q_0_diag
    Q_0_sqrt_diag = torch.sqrt(Q_0_diag)
    Q_0_sqrt_inv_diag = 1 / Q_0_sqrt_diag

    qp = bip(delta_m, Q_0_inv_diag * delta_m)

    log_det_1 = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_s, dim1=-2, dim2=-1)), dim=-1
    )
    log_det_2 = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_f, dim1=-2, dim2=-1)), dim=-1
    )
    log_det = log_det_1 + log_det_2

    tr_1_sqrt = (Q_0_sqrt_diag[None, :, None] * A) @ Psi_f
    tr_2_sqrt = Q_0_sqrt_inv_diag[None, :, None] * fast_bmm_P_f_0(
        A, Psi_f, Q_0_diag, B @ Psi_s
    )
    tr_1 = torch.sum(tr_1_sqrt.pow(2), dim=[-1, -2])
    tr_2 = torch.sum(tr_2_sqrt.pow(2), dim=[-1, -2])
    tr = L - tr_1 - tr_2

    kl = 0.5 * (qp + tr + log_det - L)

    return kl


# @torch.jit.script
def fast_bmv_P_p(M_c_p, Q_diag, v):
    """
    Multiply a vector by the predictive covariance ``P_p``.

    Parameters
    ----------
    M_c_p : torch.Tensor
        Predictive covariance factors shaped ``[batch, latents, rank]``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    v : torch.Tensor
        Vector to multiply, matching the latents dimension.

    Returns
    -------
    torch.Tensor
        Product ``P_p v``.
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
        Canonical precision factors shaped ``[batch, latents, rank]``.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    v : torch.Tensor
        Vector to multiply, matching the latents dimension.

    Returns
    -------
    torch.Tensor
        Product ``P_f v``.
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
        Inverse Cholesky of the predictive precision matrix.
    v : torch.Tensor
        Vector to multiply, matching the latents dimension.

    Returns
    -------
    torch.Tensor
        Product ``J_p v``.
    """
    Q_inv_diag = 1 / Q_diag

    u_1 = Q_inv_diag * v
    u_2 = Q_inv_diag * bmv(M_c_p, bmv(Psi_p, bmv(Psi_p.mT, bmv(M_c_p.mT, u_1))))
    u = u_1 - u_2
    return u


# @torch.jit.script
def fast_P_f_diagonal(K, Psi_f, M_c_p, Q_diag):
    """
    Extract the diagonal of the filtered covariance efficiently.

    Parameters
    ----------
    K : torch.Tensor
        Canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    torch.Tensor
        Diagonal entries of ``P_f``.
    """
    L = K.shape[-2]
    e_basis = torch.eye(L, device=K.device).view(L, L)
    p = torch.stack(
        [fast_bmv_P_f(K, Psi_f, M_c_p, Q_diag, e_basis[i])[..., i] for i in range(L)],
        dim=-1,
    )
    return p


# @torch.jit.script
def fast_bmv_P_f_0(K, Psi_f, P_p_diag, v):
    """
    Multiply a vector by the filtered covariance at the initial step.

    Parameters
    ----------
    K : torch.Tensor
        Canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    P_p_diag : torch.Tensor
        Predictive covariance diagonal entries.
    v : torch.Tensor
        Vector to multiply, matching the latents dimension.

    Returns
    -------
    torch.Tensor
        Product ``P_f v`` at time zero.
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
        Canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    P_p_diag : torch.Tensor
        Predictive covariance diagonal entries.

    Returns
    -------
    torch.Tensor
        Diagonal entries of ``P_f`` at time zero.
    """
    L = K.shape[-2]
    e_basis = torch.eye(L, device=K.device).view(L, L)
    p = torch.stack(
        [fast_bmv_P_f_0(K, Psi_f, P_p_diag, e_basis[i])[..., i] for i in range(L)],
        dim=-1,
    )
    return p


# @torch.jit.script
def fast_bmm_P_p(M_c_p, Q_diag, V):
    """
    Multiply a matrix by the predictive covariance ``P_p``.

    Parameters
    ----------
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    V : torch.Tensor
        Matrix to multiply with shape ``[batch, latents, rank]``.

    Returns
    -------
    torch.Tensor
        Product ``P_p V``.
    """
    U_1 = M_c_p @ (M_c_p.mT @ V)
    U_2 = Q_diag[None, :, None] * V
    U = U_1 + U_2
    return U


# @torch.jit.script
def fast_bmm_P_f_0(K_y, Psi_f, Q_0_diag, V):
    """
    Multiply a matrix by the filtered covariance at the initial step.

    Parameters
    ----------
    K_y : torch.Tensor
        Canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    Q_0_diag : torch.Tensor
        Initial covariance diagonal entries.
    V : torch.Tensor
        Matrix to multiply.

    Returns
    -------
    torch.Tensor
        Product ``P_f V`` at time zero.
    """
    U_1 = Q_0_diag[None, :, None] * V

    triple_bmm = K_y @ (Psi_f @ (Psi_f.mT @ (K_y.mT @ U_1)))
    U_2 = Q_0_diag[None, :, None] * triple_bmm
    U = U_1 - U_2
    return U


# @torch.jit.script
def fast_bmm_P_f(K_y, Psi_f, M_c_p, Q_diag, V):
    """
    Multiply a matrix by the filtered covariance ``P_f``.

    Parameters
    ----------
    K_y : torch.Tensor
        Forward canonical precision factors.
    Psi_f : torch.Tensor
        Inverse Cholesky of the filtered precision matrix.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    V : torch.Tensor
        Matrix to multiply.

    Returns
    -------
    torch.Tensor
        Product ``P_f V``.
    """
    U_1 = fast_bmm_P_p(M_c_p, Q_diag, V)

    W = K_y @ (Psi_f @ (Psi_f.mT @ (K_y.mT @ U_1)))
    U_2 = fast_bmm_P_p(M_c_p, Q_diag, W)
    U = U_1 - U_2
    return U


def fast_bmv_P_s(Psi_f, Psi_s, K_b, K_y, M_c_p, Q_diag, v):
    """
    Multiply a vector by the smoothed covariance ``P_s``.

    Parameters
    ----------
    Psi_f : torch.Tensor
        Inverse Cholesky of filtered precision.
    Psi_s : torch.Tensor
        Inverse Cholesky of smoothed precision.
    K_b : torch.Tensor
        Backward canonical precision factors.
    K_y : torch.Tensor
        Forward canonical precision factors.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    v : torch.Tensor
        Vector to multiply.

    Returns
    -------
    torch.Tensor
        Product ``P_s v``.
    """
    u_1 = fast_bmv_P_f(K_y, Psi_f, M_c_p, Q_diag, v)

    w = K_b @ (Psi_s @ (Psi_s.mT @ (K_b.mT @ u_1.unsqueeze(-1))))
    u_2 = fast_bmv_P_f(K_y, Psi_f, M_c_p, Q_diag, w.squeeze(-1))
    u = u_1 - u_2
    return u


def fast_bmm_P_s(Psi_f, Psi_s, K_b, K_y, M_c_p, Q_diag, V):
    """
    Multiply a matrix by the smoothed covariance ``P_s``.

    Parameters
    ----------
    Psi_f : torch.Tensor
        Inverse Cholesky of filtered precision.
    Psi_s : torch.Tensor
        Inverse Cholesky of smoothed precision.
    K_b : torch.Tensor
        Backward canonical precision factors.
    K_y : torch.Tensor
        Forward canonical precision factors.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    V : torch.Tensor
        Matrix to multiply.

    Returns
    -------
    torch.Tensor
        Product ``P_s V``.
    """
    U_1 = fast_bmm_P_f(K_y, Psi_f, M_c_p, Q_diag, V)

    W = K_b @ (Psi_s @ (Psi_s.mT @ (K_b.mT @ U_1)))
    U_2 = fast_bmm_P_f(K_y, Psi_f, M_c_p, Q_diag, W)
    U = U_1 - U_2
    return U


def fast_bmv_P_s_0(Psi_f, Psi_s, K_b, K_y, Q_0_diag, v):
    """
    Multiply a vector by the smoothed covariance at the initial step.

    Parameters
    ----------
    Psi_f : torch.Tensor
        Inverse Cholesky of filtered precision.
    Psi_s : torch.Tensor
        Inverse Cholesky of smoothed precision.
    K_b : torch.Tensor
        Backward canonical precision factors.
    K_y : torch.Tensor
        Forward canonical precision factors.
    Q_0_diag : torch.Tensor
        Initial covariance diagonal entries.
    v : torch.Tensor
        Vector to multiply.

    Returns
    -------
    torch.Tensor
        Product ``P_s v`` at time zero.
    """
    u_1 = fast_bmv_P_f_0(K_y, Psi_f, Q_0_diag, v)

    w = K_b @ (Psi_s @ (Psi_s.mT @ (K_b.mT @ u_1.unsqueeze(-1))))
    u_2 = fast_bmv_P_f_0(K_y, Psi_f, Q_0_diag, w.squeeze(-1))
    u = u_1 - u_2
    return u


def fast_bmm_P_s_0(Psi_f, Psi_s, K_b, K_y, Q_0_diag, V):
    """
    Multiply a matrix by the smoothed covariance at the initial step.

    Parameters
    ----------
    Psi_f : torch.Tensor
        Inverse Cholesky of filtered precision.
    Psi_s : torch.Tensor
        Inverse Cholesky of smoothed precision.
    K_b : torch.Tensor
        Backward canonical precision factors.
    K_y : torch.Tensor
        Forward canonical precision factors.
    Q_0_diag : torch.Tensor
        Initial covariance diagonal entries.
    V : torch.Tensor
        Matrix to multiply.

    Returns
    -------
    torch.Tensor
        Product ``P_s V`` at time zero.
    """
    U_1 = fast_bmm_P_f_0(K_y, Psi_f, Q_0_diag, V)

    W = K_b @ (Psi_s @ (Psi_s.mT @ (K_b.mT @ U_1)))
    U_2 = fast_bmm_P_f_0(K_y, Psi_f, Q_0_diag, W)
    U = U_1 - U_2
    return U


def fast_update_filtering_to_smoothing_stats_0(
    z_f, h_f, m_f, Psi_f, k_b, K_b, K_y, Q_0_diag
):
    """
    Convert filtering statistics to smoothing statistics at time zero.

    Parameters
    ----------
    z_f : torch.Tensor
        Filtered latent samples.
    h_f : torch.Tensor
        Filtered natural parameters.
    m_f : torch.Tensor
        Filtered posterior means.
    Psi_f : torch.Tensor
        Inverse Cholesky of filtered precision.
    k_b : torch.Tensor
        Backward linear statistics.
    K_b : torch.Tensor
        Backward precision factors.
    K_y : torch.Tensor
        Forward precision factors.
    Q_0_diag : torch.Tensor
        Initial covariance diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Smoothed posterior mean ``m_s``, smoothed samples ``z_s``, and inverse
        Cholesky ``Psi_s`` of the smoothed precision.
    """
    n_trials, n_latents, rank = K_b.shape
    I_r = torch.eye(rank, device=z_f.device)
    w_s = torch.randn((n_trials, rank), device=z_f.device)

    z_f_c = z_f - m_f
    P_f_K_b = fast_bmm_P_f_0(K_y, Psi_f, Q_0_diag, K_b)

    I_r_pl_triple = I_r + K_b.mT @ P_f_K_b
    I_r_pl_triple_chol = torch.linalg.cholesky(I_r_pl_triple)
    Psi_s = linalg_utils.triangular_inverse(I_r_pl_triple_chol).mT

    h_s = h_f + k_b
    m_s = fast_bmv_P_s_0(Psi_f, Psi_s, K_b, K_y, Q_0_diag, h_s)

    v_1 = bmv(K_b.mT, z_f_c) + w_s
    # z_s = m_s + z_f_c - bmv(K_b, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    unscaled_update = bmv(K_b, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    scaled_update = fast_bmv_P_f_0(K_y, Psi_f, Q_0_diag, unscaled_update)
    z_s = m_s + z_f_c - scaled_update

    return m_s, z_s, Psi_s


def fast_update_filtering_to_smoothing_stats_t(
    z_f, h_f, m_f, Psi_f, M_c_f_p, k_b, K_b, K_y, Q_diag
):
    """
    Convert filtering statistics to smoothing statistics for ``t > 0``.

    Parameters
    ----------
    z_f : torch.Tensor
        Filtered latent samples.
    h_f : torch.Tensor
        Filtered natural parameters.
    m_f : torch.Tensor
        Filtered posterior means.
    Psi_f : torch.Tensor
        Inverse Cholesky of filtered precision.
    M_c_f_p : torch.Tensor
        Predictive covariance factors from the filter.
    k_b : torch.Tensor
        Backward linear statistics.
    K_b : torch.Tensor
        Backward precision factors.
    K_y : torch.Tensor
        Forward precision factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Smoothed mean ``m_s``, smoothed samples ``z_s``, and inverse Cholesky
        ``Psi_s``.
    """
    n_trials, n_latents, rank = K_b.shape
    I_r = torch.eye(rank, device=z_f.device)
    w_s = torch.randn((n_trials, rank), device=z_f.device)

    z_f_c = z_f - m_f
    P_f_K_b = fast_bmm_P_f(K_y, Psi_f, M_c_f_p, Q_diag, K_b)

    I_r_pl_triple = I_r + K_b.mT @ P_f_K_b
    I_r_pl_triple_chol = torch.linalg.cholesky(I_r_pl_triple)
    Psi_s = linalg_utils.triangular_inverse(I_r_pl_triple_chol).mT

    h_s = h_f + k_b
    m_s = fast_bmv_P_s(Psi_f, Psi_s, K_b, K_y, M_c_f_p, Q_diag, h_s)

    v_1 = bmv(K_b.mT, z_f_c) + w_s
    # z_s = m_s + z_f_c - bmv(K_b, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    unscaled_update = bmv(K_b, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    scaled_update = fast_bmv_P_f(K_y, Psi_f, M_c_f_p, Q_diag, unscaled_update)
    z_s = m_s + z_f_c - scaled_update

    return m_s, z_s, Psi_s


# @torch.jit.script
def fast_update_step(z_p_c, h_p, k, K, w_f, M_c_p, Q_diag):
    """
    Perform the measurement update for the causal low-rank filter.

    Parameters
    ----------
    z_p_c : torch.Tensor
        Centered predictive samples with shape ``[n_samples, batch, latents]``.
    h_p : torch.Tensor
        Predictive natural parameters.
    k : torch.Tensor
        Observation natural parameters at time ``t``.
    K : torch.Tensor
        Observation precision factors at time ``t``.
    w_f : torch.Tensor
        Standard normal noise used to produce filtered samples.
    M_c_p : torch.Tensor
        Predictive covariance factors.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Posterior mean ``m``, sampled latent ``z``, inverse Cholesky ``Psi``,
        and updated natural parameters ``h``.
    """
    n_trials, n_latents, rank = K.shape
    Q_diag_sqrt = torch.sqrt(Q_diag)
    I_r = torch.eye(rank, device=z_p_c.device)

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
    return m, z, Psi, h


# @torch.jit.script
def fast_predict_step(m_theta_z_tm1, Q_diag):
    """
    Propagate the predictive distribution one step ahead.

    Parameters
    ----------
    m_theta_z_tm1 : torch.Tensor
        Dynamics samples from the previous time step with shape
        ``[batch, latents, S]``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Centered predictive samples ``z_p_c``, predictive mean ``m_p``,
        predictive natural parameters ``h_p``, covariance factors ``M_c``, and
        inverse Cholesky ``Psi_p``.
    """
    n_trials, n_latents, S = m_theta_z_tm1.shape

    sqrt_S_inv = math.sqrt(1 / S)
    Q_diag_sqrt = torch.sqrt(Q_diag)
    I_S = torch.eye(S, device=m_theta_z_tm1.device)

    w_p_1 = torch.randn([S, n_trials, S], device=m_theta_z_tm1.device)
    w_p_2 = torch.randn([S, n_trials, n_latents], device=m_theta_z_tm1.device)

    m_p = m_theta_z_tm1.mean(dim=-1)
    M_c = sqrt_S_inv * (m_theta_z_tm1 - m_p.unsqueeze(-1))

    M_c_mT_Q_inv = M_c.mT * (1 / Q_diag)
    I_pl_MmTQinvM_chol, _ = torch.linalg.cholesky_ex(I_S + M_c_mT_Q_inv @ M_c)
    Psi_p = linalg_utils.triangular_inverse(I_pl_MmTQinvM_chol).mT
    h_p = fast_bmv_P_p_inv(Q_diag, M_c, Psi_p, m_p)

    z_p_c = bmv(M_c, w_p_1) + Q_diag_sqrt * w_p_2

    return z_p_c, m_p, h_p, M_c, Psi_p


# @torch.jit.script
def fast_filter_step_t(m_theta_z_tm1, k, K, Q_diag, t_mask):
    """
    Execute a causal filtering step for ``t > 0``.

    Parameters
    ----------
    m_theta_z_tm1 : torch.Tensor
        Dynamics samples from the previous time step.
    k : torch.Tensor
        Observation natural parameters at time ``t``.
    K : torch.Tensor
        Observation precision factors at time ``t``.
    Q_diag : torch.Tensor
        Process noise diagonal entries.
    t_mask : bool
        When True, skip the measurement update.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Filtered samples ``z_f``, posterior mean ``m_f``, predictive mean ``m_p``,
        covariance factors ``M_c_p``, filtered inverse Cholesky ``Psi_f``,
        predictive inverse Cholesky ``Psi_p``, and filtered natural parameters ``h_f``.
    """
    n_trials, n_latents, rank = K.shape
    n_samples = m_theta_z_tm1.shape[-1]
    batch_sz = [n_trials]

    w_f = torch.randn([n_samples] + batch_sz + [rank], device=m_theta_z_tm1.device)
    z_p_c, m_p, h_p, M_c_p, Psi_p = fast_predict_step(m_theta_z_tm1, Q_diag)
    m_f, z_f, Psi_f, h_f = fast_update_step(z_p_c, h_p, k, K, w_f, M_c_p, Q_diag)

    return z_f, m_f, m_p, M_c_p, Psi_f, Psi_p, h_f


# @torch.jit.script
def fast_update_step_0(z_p_c, h_p, k, K, w_f, P_p_diag):
    """
    Perform the measurement update for the initial time step.

    Parameters
    ----------
    z_p_c : torch.Tensor
        Centered predictive samples at time zero.
    h_p : torch.Tensor
        Predictive natural parameters.
    k : torch.Tensor
        Observation natural parameters at time zero.
    K : torch.Tensor
        Observation precision factors at time zero.
    w_f : torch.Tensor
        Standard normal noise used to sample filtered latents.
    P_p_diag : torch.Tensor
        Predictive covariance diagonal entries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Posterior mean ``m``, sampled latent ``z``, inverse Cholesky ``Psi``,
        and updated natural parameters ``h``.
    """
    n_trials, n_latents, rank = K.shape
    I_r = torch.eye(rank, device=K.device)

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

    return m, z, Psi, h


# @torch.jit.script
def fast_filter_step_0(
    m_0: torch.Tensor,
    k: torch.Tensor,
    K: torch.Tensor,
    P_p_diag: torch.Tensor,
    n_samples: int,
):
    """
    Execute the initial causal filtering step.

    Parameters
    ----------
    m_0 : torch.Tensor
        Initial mean vector.
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
        Filtered samples ``z_f``, posterior mean ``m_f``, predictive mean ``m_p``,
        inverse Cholesky ``Psi_f``, and filtered natural parameters ``h_f``.
    """
    n_trials, n_latents, rank = K.shape
    batch_sz = [n_trials]
    w_p = torch.randn([n_samples] + batch_sz + [n_latents], device=K.device)

    z_p_c = torch.sqrt(P_p_diag) * w_p
    J_p_diag = 1 / P_p_diag
    m_p = m_0 * torch.ones(batch_sz + [n_latents], device=m_0.device)
    h_p = J_p_diag * m_p

    w_f = torch.randn([n_samples] + batch_sz + [rank], device=K.device)
    m_f, z_f, Psi_f, h_f = fast_update_step_0(z_p_c, h_p, k, K, w_f, P_p_diag)

    return z_f, m_f, m_p, Psi_f, h_f
