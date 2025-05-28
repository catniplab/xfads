import torch
from torch import nn
import torch.nn.functional as Fn
from .. import prob_utils
from ..linalg_utils import bmv, bip, chol_bmv_solve


class SVAE(nn.Module):
    def __init__(self, dynamics_mod, likelihood_pdf, initial_c_pdf, local_encoder):
        super(SVAE, self).__init__()

        self.dynamics_mod = dynamics_mod
        self.local_encoder = local_encoder
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf

    @torch.jit.export
    def forward(
        self,
        y,
        n_samples: int,
        p_mask_y_in: float = 0.0,
        p_mask_apb: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
    ):
        z_s, stats = self.smooth_1_to_T(
            y,
            n_samples,
            p_mask_apb=p_mask_apb,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_b=p_mask_b,
            get_kl=True,
        )

        ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)

        loss = stats["kl"] - ell
        loss = loss.sum(dim=-1).mean()
        return loss, z_s, stats

    def smooth_1_to_T(
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
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins)))
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons))
        )
        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        F = self.dynamics_mod.mean_fn.weight
        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        Q_0_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
        m_0 = self.initial_c_pdf.m_0

        m_f, P_f, m_p, P_p = prob_utils.kalman_information_filter(
            k_y, K_y @ K_y.mT, F, Q_diag, m_0, Q_0_diag
        )
        m, P = prob_utils.rts_smoother(m_p, P_p, m_f, P_f, F)

        kl = self._get_kl(m, m_f, m_p, P, P_p, k_y, K_y)
        z_s = m + bmv(
            torch.linalg.cholesky(P),
            torch.randn(
                (n_samples, n_trials, n_time_bins, m.shape[-1]), device=m.device
            ),
        )

        stats = {}
        stats["m"] = m
        stats["P"] = P
        stats["m_p"] = m_p
        stats["P_p"] = P_p
        stats["m_f"] = m_f
        stats["P_f"] = P_f
        stats["kl"] = kl

        return z_s, stats

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

    def _get_kl(self, m_s, m_f, m_p, P_s, P_p, a, A):
        m_0 = self.initial_c_pdf.m_0
        Q_0_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
        # Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        # F = self.dynamics_mod.mean_fn.weight

        kl_1 = alt_kl_step_0(
            m_s[:, 0], m_f[:, 0], m_0, P_s[:, 0], Q_0_diag, a[:, 0], A[:, 0]
        )
        kl_2_T = alt_kl_step_t(
            m_s[:, 1:],
            m_f[:, 1:],
            m_p[:, 1:],
            P_s[:, 1:],
            P_p[:, 1:],
            a[:, 1:],
            A[:, 1:],
        )

        # m_s_p_2_T = bmv(F, m_s[:, :-1])
        # P_s_p_2_T = F @ P_s[:, :-1] @ F.mT + torch.diag(Q_diag)
        # kl_1 = prob_utils.kl_dense_gaussian_mean_covariance(m_s[:, 0], P_s[:, 0], m_0, torch.diag(Q_0_diag))
        # kl_2_T = prob_utils.kl_dense_gaussian_mean_covariance(m_s[:, 1:], P_s[:, 1:], m_s_p_2_T, P_s_p_2_T)

        kl = torch.concat([kl_1.unsqueeze(1), kl_2_T], dim=1)
        return kl


def alt_kl_step_t(m_s, m_f, m_p, P_s, P_p, a, A):
    P_sA = P_s @ A
    P_pA = P_p @ A
    P_p_chol = torch.linalg.cholesky(P_p)

    AmTP_sA = A.mT @ P_sA
    AmTP_pA = A.mT @ P_pA
    AmTm_s = bmv(A.mT, m_s)
    AmTm_f = bmv(A.mT, m_f)
    P_p_inv_m_f = chol_bmv_solve(P_p_chol, m_f)
    P_p_inv_m_p = chol_bmv_solve(P_p_chol, m_p)
    logdet_triple = torch.logdet(torch.eye(A.shape[-1], device=m_s.device) + AmTP_pA)

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


def alt_kl_step_0(m_s, m_f, m_0, P_s, Q_0_diag, a, A):
    P_sA = P_s @ A
    P_pA = A * Q_0_diag[:, None]

    AmTP_sA = A.mT @ P_sA
    AmTP_pA = A.mT @ P_pA
    AmTm_s = bmv(A.mT, m_s)
    AmTm_f = bmv(A.mT, m_f)
    P_p_inv_m_f = (1 / Q_0_diag) * m_f
    P_p_inv_m_p = (1 / Q_0_diag) * m_0
    logdet_triple = torch.logdet(torch.eye(A.shape[-1], device=m_s.device) + AmTP_pA)

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
