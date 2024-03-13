import torch
import torch.nn as nn
import dev.utils as utils


class DenseGaussianDynamics(nn.Module):
    def __init__(self, mean_fn, n_latents, Q_diag, device='cpu', fix_Q=False):
        super(DenseGaussianDynamics, self).__init__()
        self.device = device

        self.mean_fn = mean_fn
        self.n_latents = n_latents
        self.n_nat_params = n_latents + n_latents**2

        if fix_Q:
            self.log_Q = utils.softplus_inv(Q_diag)
        else:
            self.log_Q = torch.nn.Parameter(utils.softplus_inv(Q_diag))


class DenseGaussianInitialCondition(nn.Module):
    def __init__(self, n_latents, m_0, Q_0_diag, device='cpu', fix_Q_0=False):
        super(DenseGaussianInitialCondition, self).__init__()
        self.device = device
        self.n_latents = n_latents
        self.m_0 = torch.nn.Parameter(m_0).to(self.device)

        if fix_Q_0:
            self.log_Q_0 = utils.softplus_inv(Q_0_diag)
        else:
            self.log_Q_0 = torch.nn.Parameter(utils.softplus_inv(Q_0_diag)).to(device)
