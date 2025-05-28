import torch
from torch import nn
import torch.nn.functional as Fn
from .. import prob_utils


class dKF(nn.Module):
    def __init__(
        self,
        dynamics_mod,
        likelihood_pdf,
        initial_c_pdf,
        local_encoder,
        backward_encoder,
        init_c_encoder,
    ):
        super().__init__()

        self.dynamics_mod = dynamics_mod
        self.likelihood_pdf = likelihood_pdf
        self.initial_c_pdf = initial_c_pdf
        self.local_encoder = local_encoder
        self.backward_encoder = backward_encoder
        self.init_c_encoder = init_c_encoder  # n_hidden_backward -> 2 * n_latents

    def forward(self, y, n_samples, **kwargs):
        n_trials, n_time_bins, n_neurons = y.shape
        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)

        z = []
        kl = []
        h = self.backward_encoder(y)

        for t in range(n_time_bins):
            if t == 0:
                m_0 = self.initial_c_pdf.m_0
                Q_0_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
                m_t, P_diag_t = self.init_c_encoder(h[:, 0])
                z_t = m_t + torch.sqrt(P_diag_t) * torch.randn(
                    (n_samples, n_trials, m_0.shape[-1]), device=y.device
                )
                kl_t = prob_utils.kl_diagonal_gaussian_canon(
                    m_t, P_diag_t, m_0, Q_0_diag
                )
            else:
                m_t, P_diag_t = self.local_encoder(
                    h[None, :, t].expand(z[t - 1].shape[0], n_trials, h.shape[-1]),
                    z[t - 1],
                )
                z_t = m_t + torch.sqrt(P_diag_t) * torch.randn_like(
                    z[t - 1], device=y.device
                )
                kl_t = prob_utils.kl_diagonal_gaussian_canon(
                    m_t, P_diag_t, self.dynamics_mod.mean_fn(z[t - 1]), Q_diag
                )
                kl_t = kl_t.mean(dim=0)

            z.append(z_t)
            kl.append(kl_t)

        z = torch.stack(z, dim=2)
        kl = torch.stack(kl, dim=1)
        ell = self.likelihood_pdf.get_ell(y, z).mean(dim=0)
        loss = (kl - ell).sum(dim=-1).mean()

        stats = {}
        stats["m"] = z.mean(dim=0)

        return loss, z, stats

    def predict_forward(self, z_tm1: torch.Tensor, n_bins: int):
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
