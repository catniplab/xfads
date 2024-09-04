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
            self.log_R = utils.softplus_inv(R_diag)
        else:
            self.log_R = torch.nn.Parameter(utils.softplus_inv(R_diag))

    def get_ell(self, y, z):
        mean = self.readout_fn(z)
        cov = Fn.softplus(self.log_R)
        log_prob = -0.5 * ((y - mean)**2 / cov + torch.log(cov) + math.log(2 * math.pi))
        log_p_y = log_prob.sum(dim=-1)
        return log_p_y


@apply_memory_cleanup
class PoissonLikelihood(nn.Module):
    def __init__(self, readout_fn, n_neurons, delta, device='cpu', p_mask=0.0):
        super(PoissonLikelihood, self).__init__()
        self.delta = delta
        self.device = device
        self.n_neurons = n_neurons
        self.readout_fn = readout_fn

    def get_ell(self, y, z, reduce_neuron_dim=True):
        log_exp = math.log(self.delta) + self.readout_fn(z) # C @ z
        log_p_y = -torch.nn.functional.poisson_nll_loss(log_exp, y, full=True, reduction='none')

        if reduce_neuron_dim:
            return log_p_y.sum(dim=-1)
        else:
            return log_p_y



