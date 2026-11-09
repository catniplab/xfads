import math
import torch
import torch.nn as nn
import xfads.utils as utils
import torch.nn.functional as Fn

from xfads.decorators import *


@apply_memory_cleanup
class GaussianLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, R_diag, device='cpu', fix_R=False):
        super(GaussianLikelihood, self).__init__()

        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

        if fix_R:
            self.register_buffer('R_diag', R_diag)
            self.log_R = utils.softplus_inv(R_diag)
        else:
            self.log_R = torch.nn.Parameter(utils.softplus_inv(R_diag))

    def get_ell(self, y, z, reduce_neuron_dim=True):
        # missing entries (NaN) add no term; their predictions are replaced before any arithmetic,
        # so an overflow there cannot turn into NaN (inf * 0) in the loss or its gradient
        obs = ~y.isnan()
        mean = torch.where(obs, self.readout_fn(z), 0.)
        cov = Fn.softplus(self.log_R)
        log_prob = -0.5 * ((torch.where(obs, y, 0.) - mean)**2 / cov + torch.log(cov) + math.log(2 * math.pi)) * obs

        if reduce_neuron_dim:
            return log_prob.sum(dim=-1)
        else:
            return log_prob


@apply_memory_cleanup
class PoissonLikelihood(nn.Module):
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


class BernoulliLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, device='cpu', p_mask=0.0):
        super(BernoulliLikelihood, self).__init__()
        self.device = device
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

    def get_ell(self, y, z, reduce_neuron_dim=True):
        obs = ~y.isnan()  # missing entries (NaN) add no term
        logits = self.readout_fn(z)
        y = torch.where(obs, y, 0.).expand([logits.shape[0]] + list(y.shape))
        log_p_y = -torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * obs

        if reduce_neuron_dim:
            return log_p_y.sum(dim=-1)
        else:
            return log_p_y


