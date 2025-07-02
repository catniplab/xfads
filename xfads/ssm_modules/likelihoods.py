import math
import torch
from torch import nn
import torch.nn.functional as Fn
from .. import utils


class GaussianLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, R_diag, fix_R=False):
        super().__init__()

        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

        if fix_R:
            self.register_buffer('R_diag', R_diag)
            self.log_R = utils.softplus_inv(R_diag)
        else:
            self.log_R = torch.nn.Parameter(utils.softplus_inv(R_diag))

    def get_ell(self, y, z):
        mean = self.readout_fn(z)
        cov = Fn.softplus(self.log_R.to(z))
        log_prob = -0.5 * (
            (y - mean) ** 2 / cov + torch.log(cov) + math.log(2 * math.pi)
        )
        log_p_y = log_prob.sum(dim=-1)
        return log_p_y


class PoissonLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, delta, p_mask=0.0):
        super().__init__()
        self.delta = delta
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

    def get_ell(self, y, z, reduce_neuron_dim=True):
        log_exp = math.log(self.delta) + self.readout_fn(z)  # C @ z
        log_p_y = -torch.nn.functional.poisson_nll_loss(
            log_exp, y, full=True, reduction="none"
        )

        if reduce_neuron_dim:
            return log_p_y.sum(dim=-1)
        else:
            return log_p_y


class BernoulliLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, p_mask=0.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

    def get_ell(self, y, z, reduce_neuron_dim=True):
        logits = self.readout_fn(z)
        log_p_y = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y.expand([logits.shape[0]] + list(y.shape)), reduction="none"
        )

        if reduce_neuron_dim:
            return log_p_y.sum(dim=-1)
        else:
            return log_p_y
