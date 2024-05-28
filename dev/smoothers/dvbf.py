import torch
import torch.nn as nn
import torch.nn.functional as Fn


class dVBF(nn.Module):
    def __init__(self, dynamics_mod, likelihood_pdf, initial_c_pdf,
                 encoder, device='cpu'):
        super(dVBF, self).__init__()
        self.device = device

        self.dynamics_mod = dynamics_mod
        self.likelihood_pdf = likelihood_pdf
        self.initial_c_pdf = initial_c_pdf
        self.encoder = encoder

    def forward(self, y, n_samples, p_mask_a=0.0, **kwargs):

        n_trials, n_time_bins, n_neurons = y.shape
        Q_sqrt = torch.sqrt(Fn.softplus(self.dynamics_mod.log_Q))
        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=y.device))

        z = []
        m, P_diag = self.encoder(y)
        m = m * t_mask_a[..., None]
        P_diag = P_diag * t_mask_a[..., None] + (1 - t_mask_a[..., None])

        w = m + torch.sqrt(P_diag) * torch.randn((n_samples, n_trials, n_time_bins, Q_sqrt.shape[-1]), device=y.device)

        for t in range(n_time_bins):
            if t == 0:
                Q_0 = Fn.softplus(self.initial_c_pdf.log_Q_0)
                z_t = self.initial_c_pdf.m_0 + torch.sqrt(Q_0) * w[:, :, 0]
            else:
                z_t = self.dynamics_mod.mean_fn(z[t-1]) + Q_sqrt * w[:, :, t]

            z.append(z_t)

        z = torch.stack(z, dim=2)
        ell = self.likelihood_pdf.get_ell(y, z).mean(dim=0)
        kl = 0.5 * (P_diag + m**2 - torch.log(P_diag) - 1).sum(dim=-1)
        loss = (kl - ell).sum(dim=-1).mean()

        stats = {}
        stats['m'] = z.mean(dim=0)
        return loss, z, stats

    def predict_forward(self,
                        z_tm1: torch.Tensor,
                        n_bins: int):

        z_forward = []
        Q_sqrt = torch.sqrt(Fn.softplus(self.dynamics_mod.log_Q))

        for t in range(n_bins):
            if t == 0:
                z_t = self.dynamics_mod.mean_fn(z_tm1) + Q_sqrt * torch.randn_like(z_tm1, device=z_tm1.device)
            else:
                z_t = self.dynamics_mod.mean_fn(z_forward[t-1]) + Q_sqrt * torch.randn_like(z_forward[t-1], device=z_tm1.device)

            z_forward.append(z_t)

        z_forward = torch.stack(z_forward, dim=2)
        return z_forward
