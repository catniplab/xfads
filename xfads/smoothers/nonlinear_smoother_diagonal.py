import torch
from torch import nn
import torch.nn.functional as Fn
from ..linalg_utils import bop, bip, chol_bmv_solve


class DiagonalNonlinearStateSpaceModel(nn.Module):
    def __init__(
        self,
        dynamics_mod,
        likelihood_pdf,
        initial_c_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
    ):
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
    ):
        z_s, stats = self.fast_smooth_1_to_T(
            y,
            n_samples,
            p_mask_apb=p_mask_apb,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_b=p_mask_b,
            get_kl=True,
        )

        ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)
        kl = trajectory_kl_diagonal_dense(
            stats["m_f"], stats["P_f"], stats["m_p"], stats["P_p_chol"]
        )

        loss = kl - ell
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
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, 1, n_neurons))
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
    ):
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins)))
        t_mask_b = torch.bernoulli((1 - p_mask_b) * torch.ones((n_trials, n_time_bins)))
        t_mask_apb = torch.bernoulli(
            (1 - p_mask_apb) * torch.ones((n_trials, n_time_bins))
        )
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons))
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b
        K_b = t_mask_b[..., None] * K_b

        k = k_b + k_y
        K = K_b + K_y
        k = t_mask_apb[..., None] * k
        K = t_mask_apb[..., None] * K

        z_s, stats = self.nl_filter(k, K, n_samples, get_kl=get_kl, get_v=get_v)
        stats["t_mask_y_in"] = t_mask_y_in

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


class NonlinearFilter(nn.Module):
    def __init__(self, dynamics_mod, initial_c_pdf):
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
    ):
        # mask data, 0: data available, 1: data missing
        n_trials, n_time_bins, n_latents = K.shape

        m_f = []
        P_f = []
        m_p = []
        P_p_chol = []

        z_f = []
        stats = {}

        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)

        for t in range(n_time_bins):
            if t == 0:
                m_0 = self.initial_c_pdf.m_0
                P_p_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
                z_f_t, m_f_t, m_p_t, P_f_t, P_p_chol_t = filter_step_0(
                    m_0, k[:, 0], K[:, 0], P_p_diag, n_samples
                )

            else:
                m_fn_z_tm1 = self.dynamics_mod.mean_fn(z_f[t - 1]).movedim(0, -1)
                z_f_t, m_f_t, m_p_t, P_f_t, P_p_chol_t = filter_step_t(
                    m_fn_z_tm1, k[:, t], K[:, t], Q_diag
                )

            m_f.append(m_f_t)
            P_f.append(P_f_t)
            m_p.append(m_p_t)
            z_f.append(z_f_t)
            P_p_chol.append(P_p_chol_t)

        z_f = torch.stack(z_f, dim=2)
        stats["m_f"] = torch.stack(m_f, dim=1)
        stats["m_p"] = torch.stack(m_p, dim=1)
        stats["P_f"] = torch.stack(P_f, dim=1)
        stats["P_p_chol"] = torch.stack(P_p_chol, dim=1)

        return z_f, stats


def trajectory_kl(m_f, P_f, m_p, P_p):
    kl = 0.5 * (torch.log(P_p / P_f) + (P_f + (m_f - m_p)) / P_f - 1)
    return kl.sum(dim=-1)


def trajectory_kl_diagonal_dense(m_f, P_f, m_p, P_p_chol):
    L = m_p.shape[-1]

    diff = m_p - m_f
    qp = bip(diff, chol_bmv_solve(P_p_chol, diff))
    tr = torch.sum(
        torch.diagonal(torch.cholesky_inverse(P_p_chol), dim1=-2, dim2=-1) * P_f, dim=-1
    )
    logdet_p = 2 * torch.sum(
        torch.log(torch.diagonal(P_p_chol, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )
    logdet_f = torch.sum(torch.log(P_f), dim=-1)

    kl = 0.5 * (tr + qp + logdet_p - logdet_f - L)
    return kl


def predict_step_t(m_theta_z_tm1, Q_diag):
    M = -0.5 * (torch.diag(Q_diag) + bop(m_theta_z_tm1, m_theta_z_tm1))

    m_p = m_theta_z_tm1.mean(dim=0)
    M_p = M.mean(dim=0)
    P_p = -2 * M_p - bop(m_p, m_p)
    return m_p, P_p


def filter_step_t(m_theta_z_tm1, k, K, Q_diag):
    n_trials, n_latents = K.shape
    n_samples = m_theta_z_tm1.shape[-1]
    batch_sz = [n_trials]

    w_f = torch.randn([n_samples] + batch_sz + [n_latents])
    m_p, P_p = predict_step_t(m_theta_z_tm1.movedim(-1, 0), Q_diag)
    P_p_chol = torch.linalg.cholesky(P_p)

    # h_p = chol_bmv_solve(P_p_chol, m_p)
    h_p = (1 / torch.diagonal(P_p, dim1=-2, dim2=-1)) * m_p
    h_f = h_p + k

    J_f = (1 / torch.diagonal(P_p, dim1=-2, dim2=-1)) + K
    P_f = 1 / J_f
    m_f = P_f * h_f

    z_f = m_f + torch.sqrt(P_f) * w_f

    return z_f, m_f, m_p, P_f, P_p_chol


# @torch.jit.script
def filter_step_0(
    m_0: torch.Tensor,
    k: torch.Tensor,
    K: torch.Tensor,
    P_0_diag: torch.Tensor,
    n_samples: int,
):
    n_trials, n_latents = K.shape
    batch_sz = [n_trials]

    J_0_diag = 1 / P_0_diag
    h_0 = J_0_diag * m_0
    J_f = J_0_diag + K
    P_f = 1 / J_f

    h_f = h_0 + k
    m_f = P_f * h_f

    w_f = torch.randn([n_samples] + batch_sz + [n_latents])
    z_f = m_f + torch.sqrt(P_f) * w_f

    m_p = m_0 * torch.ones_like(m_f)
    P_p_chol = torch.diag(torch.sqrt(P_0_diag)) * torch.ones(
        list(m_f.shape) + [n_latents]
    )

    return z_f, m_f, m_p, P_f, P_p_chol
