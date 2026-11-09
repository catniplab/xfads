import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import xfads.utils as utils
import torch.nn.functional as Fn

from xfads.decorators import *


class Likelihood(nn.Module, ABC):
    """Observation likelihood interface consumed by the XFADS smoothers.

    Subclasses implement :meth:`get_ell` and register trainable parameters,
    buffers, and readout modules through the usual ``nn.Module`` mechanisms.
    Call ``super().__init__()`` in subclass constructors.
    Smoothers call ``get_ell`` directly; implementing ``forward`` is not required.

    This interface describes the common likelihood operation only.
    Some specialized smoothers also require access to ``readout_fn``;
    those requirements depend on the smoother being used.
    """

    @abstractmethod
    def get_ell(self, y: torch.Tensor, z: torch.Tensor,
                reduce_neuron_dim: bool = True,
                full: bool = False) -> torch.Tensor:
        """Evaluate observation log likelihoods at the supplied latent states.

        Parameters
        ----------
        y : torch.Tensor
            Observations with shape ``(n_trials, n_time_bins, n_neurons)``.
            Values must belong to the observation model's support.
            NaN entries denote missing observations and contribute zero.
        z : torch.Tensor
            Latent samples with shape
            ``(n_samples, n_trials, n_time_bins, n_latents)``.
            The readout must preserve the leading dimensions and produce
            ``n_neurons`` observation parameters per latent state.
            Observations are broadcast over the leading sample dimension.
            Inputs and module parameters must have compatible devices and
            dtypes.
        reduce_neuron_dim : bool, optional
            Sum over neurons when True (the default).
            When False, return separate terms for each neuron, as required
            by coordinated dropout.
        full : bool, optional
            Skip observation-only normalization work when False (the default).
            Set True to include these terms for absolute likelihood reporting.
            Every term depending on latent states or trainable likelihood
            parameters is retained in both modes.
            The built-in likelihoods return normalized log probabilities or
            densities in full mode.

        Returns
        -------
        torch.Tensor
            Conditional log likelihoods with shape
            ``(n_samples, n_trials, n_time_bins)`` when summing over neurons,
            or ``(n_samples, n_trials, n_time_bins, n_neurons)`` otherwise.
            Keep sample, trial, and time dimensions intact, including when
            their size is one.
            Return log likelihoods, rather than negative log likelihoods.
            Preserve the autograd graph with respect to latent states and
            trainable likelihood parameters.

        Notes
        -----
        Despite the name, this method evaluates each supplied latent sample;
        it does not compute an expectation or draw additional samples.
        Smoothers compute the Monte Carlo expectation by averaging dimension 0.
        No averaging over trials or summation over time is performed here.

        Normalization convention: retain every term that depends on ``z`` or
        any trainable likelihood parameter, including parameter-dependent
        normalizers.
        With ``full=False``, omit only additive terms depending on ``y`` and
        fixed constants, and document the omitted terms in each implementation.
        Omitting these terms preserves gradients with respect to latent states
        and trainable likelihood parameters, but shifts likelihood
        and ELBO values; the returned value is then not a normalized log
        probability or density and may not be comparable across models.
        Gradients with respect to observations can differ between modes.
        """
        raise NotImplementedError


@apply_memory_cleanup
class GaussianLikelihood(Likelihood):
    """Diagonal Gaussian log density with optional constant normalization.

    ``full=True`` includes all normalization terms.
    The default ``full=False`` omits the constant half-log-two-pi term per
    observed neuron.
    The variance-dependent normalizer is retained even when variance is fixed.
    ``R_diag`` specifies observation variances, not standard deviations.
    ``reduce_neuron_dim=False`` preserves the final neuron dimension.
    """

    def __init__(self, readout_fn, n_neurons, R_diag, device='cpu', fix_R=False):
        super(GaussianLikelihood, self).__init__()

        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

        if fix_R:
            self.register_buffer('R_diag', R_diag)
            self.log_R = utils.softplus_inv(R_diag)
        else:
            self.log_R = torch.nn.Parameter(utils.softplus_inv(R_diag))

    def get_ell(self, y, z, reduce_neuron_dim=True, full=False):
        # missing entries (NaN) add no term; their predictions are replaced before any arithmetic,
        # so an overflow there cannot turn into NaN (inf * 0) in the loss or its gradient
        obs = ~y.isnan()
        mean = torch.where(obs, self.readout_fn(z), 0.)
        cov = Fn.softplus(self.log_R)
        log_prob = (torch.where(obs, y, 0.) - mean)**2 / cov + torch.log(cov)
        if full:
            log_prob = log_prob + math.log(2 * math.pi)
        log_prob = -0.5 * log_prob * obs

        if reduce_neuron_dim:
            return log_prob.sum(dim=-1)
        else:
            return log_prob


@apply_memory_cleanup
class PoissonLikelihood(Likelihood):
    """Poisson log mass with an optional exact log-factorial term.

    ``full=True`` includes the observation-only log-factorial term using
    ``torch.lgamma(y + 1)``, matching ``torch.distributions.Poisson.log_prob``.
    The default ``full=False`` skips computing this term.
    Both modes use ``poisson_nll_loss(full=False)``; PyTorch's approximate
    full-mode factorial term is not used.
    ``delta`` is the bin duration; the readout produces log rates per unit time.
    ``reduce_neuron_dim=False`` preserves the final neuron dimension.
    """

    def __init__(self, readout_fn, n_neurons, delta, device='cpu', p_mask=0.0):
        super(PoissonLikelihood, self).__init__()
        self.delta = delta
        self.device = device
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

    def get_ell(self, y, z, reduce_neuron_dim=True, full=False):
        # missing entries (NaN) add no term; their predictions are replaced before any arithmetic,
        # so an overflow there cannot turn into NaN (inf * 0) in the loss or its gradient
        obs = ~y.isnan()
        y = torch.where(obs, y, 0.)
        log_exp = torch.where(obs, math.log(self.delta) + self.readout_fn(z), 0.) # C @ z
        # keep full=False when training: -log(y!) is constant in every parameter, so gradients and
        # bits/spike do not change. full=True adds the exact -log(y!) for reporting a normalized
        # log-likelihood; torch's own full=True would use Stirling's approximation, inexact and with
        # a NaN backward w.r.t. y (only y) at y = 0
        log_p_y = -torch.nn.functional.poisson_nll_loss(log_exp, y, full=False, reduction='none')
        if full:
            log_p_y = log_p_y - torch.lgamma(y + 1)
        log_p_y = log_p_y * obs

        if reduce_neuron_dim:
            return log_p_y.sum(dim=-1)
        else:
            return log_p_y


class BernoulliLikelihood(Likelihood):
    """Fully normalized Bernoulli log mass for binary observations.

    The readout produces logits, and no normalization terms are omitted.
    Binary observations have no removable observation-only term, so ``full``
    has no effect on either values or computation.
    ``reduce_neuron_dim=False`` preserves the final neuron dimension.
    """

    def __init__(self, readout_fn, n_neurons, device='cpu', p_mask=0.0):
        super(BernoulliLikelihood, self).__init__()
        self.device = device
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

    def get_ell(self, y, z, reduce_neuron_dim=True, full=False):
        obs = ~y.isnan()  # missing entries (NaN) add no term
        logits = self.readout_fn(z)
        y = torch.where(obs, y, 0.).expand([logits.shape[0]] + list(y.shape))
        log_p_y = -torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * obs

        if reduce_neuron_dim:
            return log_p_y.sum(dim=-1)
        else:
            return log_p_y


