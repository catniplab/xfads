import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import dev.linalg_utils as linalg_utils
from dev.linalg_utils import bmv, bip, chol_bmv_solve


class LowRankNonlinearStateSpaceModel(nn.Module):
    def __init__(self, dynamics_mod, likelihood_pdf,
                 initial_c_pdf, backward_encoder, local_encoder, nl_filter, device='cpu'):
        super(LowRankNonlinearStateSpaceModel, self).__init__()

        self.device = device
        self.nl_filter = nl_filter
        self.dynamics_mod = dynamics_mod
        self.local_encoder = local_encoder
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf
        self.backward_encoder = backward_encoder

    @torch.jit.export
    def forward(self,
                y,
                n_samples: int,
                p_mask_y_in: float = 0.0,
                p_mask_a: float = 0.0,
                p_mask_b: float = 0.0,
                p_mask_apb: float = 0.0):

        z_s, stats = self.fast_smooth_1_to_T(y, n_samples, p_mask_y_in=p_mask_y_in,
                                             p_mask_a=p_mask_a, p_mask_b=p_mask_b, get_kl=True)

        ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)
        loss = stats['kl'] - ell
        loss = loss.sum(dim=-1).mean()
        return loss, z_s, stats


    @torch.jit.export
    def fast_smooth_1_to_T(self,
                           y,
                           n_samples: int,
                           p_mask_a: float = 0.0,
                           p_mask_y_in: float = 0.0,
                           p_mask_b: float = 0.0,
                           get_kl: bool = False,
                           get_v: bool = False):

        device = y.device
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=device))
        t_mask_b = torch.bernoulli((1 - p_mask_b) * torch.ones((n_trials, n_time_bins), device=device))
        t_mask_y_in = torch.bernoulli((1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons), device=device))

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y / (1 - p_mask_a)
        K_y = t_mask_a[..., None, None] * K_y / (1 - p_mask_a)

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b / (1 - p_mask_b)
        K_b = t_mask_b[..., None, None] * K_b / (1 - p_mask_b)

        z_s, stats = self.nl_filter(k_y, K_y, k_b, K_b, n_samples, get_kl=get_kl)
        return z_s, stats


class LrSSMcoBPS(LowRankNonlinearStateSpaceModel):
    def __init__(self, dynamics_mod, likelihood_pdf,
                 initial_c_pdf, backward_encoder, local_encoder, nl_filter, device='cpu'):
        super(LowRankNonlinearStateSpaceModel, self).__init__()

        self.device = device
        self.nl_filter = nl_filter
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf
        self.local_encoder = local_encoder
        self.likelihood_pdf = likelihood_pdf
        self.backward_encoder = backward_encoder

    @torch.jit.export
    def forward(self,
                y_obs,
                y_enc,
                n_samples: int,
                p_mask_dyn: float=0.0,
                p_mask_obs: float=0.0):

        z_s, stats = self.fast_smooth_1_to_T(y_enc, n_samples, p_mask_dyn, get_kl=True)
        ell = self.likelihood_pdf.get_ell(y_obs, z_s, p_mask_obs).mean(dim=0)
        loss = stats['kl'] - ell / (1 - p_mask_obs)
        loss = loss.sum(dim=-1).mean()
        print(stats['kl'].min())
        return loss, z_s, stats

    @torch.jit.export
    def predict(self,
                y_enc,
                n_samples: int):

        z_s, stats = self.fast_smooth_1_to_T(y_enc, n_samples, p_mask=0.0, get_kl=True)

        # expected log rate or log expected rate
        log_rate_hat = math.log(self.likelihood_pdf.delta) +  self.likelihood_pdf.readout_fn(stats['m_s'])
        stats['log_rate'] = log_rate_hat
        return z_s, stats


class NonlinearFilter(nn.Module):
    def __init__(self, dynamics_mod, initial_c_pdf, device):
        super(NonlinearFilter, self).__init__()

        self.device = device
        self.dynamics_mod = dynamics_mod
        self.initial_c_pdf = initial_c_pdf

    def forward(self,
                k_y: torch.Tensor,
                K_y: torch.Tensor,
                k_b: torch.Tensor,
                K_b: torch.Tensor,
                n_samples: int,
                get_kl: bool=False,
                p_mask: float=0.0):

        # mask data, 0: data available, 1: data missing
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
            if t == 0:
                m_0 = self.initial_c_pdf.m_0
                Q_0_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
                Psi_p_t = torch.zeros((n_trials, n_samples, n_samples), device=k_y.device)

                z_f_t, m_f_t, m_p_t, Psi_f_t, h_f_t = fast_filter_step_0(m_0, k_y[:, 0], K_y[:, 0], Q_0_diag, n_samples)
                m_s_t, z_s_t, Psi_s_t = fast_update_filtering_to_smoothing_stats_0(z_f_t, h_f_t, m_f_t, Psi_f_t, k_b[:, t], K_b[:, t], K_y[:, t], Q_0_diag)

                kl_t = low_rank_kl_step_0(m_s_t, m_0, Q_0_diag, Q_diag, K_y[:, 0], K_b[:, 0], Psi_f_t, Psi_s_t)

            else:
                m_fn_z_f_tm1 = self.dynamics_mod.mean_fn(z_f[t-1]).movedim(0, -1)
                m_fn_z_s_tm1 = self.dynamics_mod.mean_fn(z_s[t-1]).movedim(0, -1)
                z_f_t, m_f_t, m_p_t, M_p_c_t, Psi_f_t, Psi_p_t, h_f_t = fast_filter_step_t(m_fn_z_f_tm1, k_y[:, t], K_y[:, t], Q_diag, torch.tensor(False))
                m_s_t, z_s_t, Psi_s_t = fast_update_filtering_to_smoothing_stats_t(z_f_t, h_f_t, m_f_t, Psi_f_t,
                                                                                   M_p_c_t, k_b[:, t], K_b[:, t],
                                                                                   K_y[:, t], Q_diag)

                _, m_s_p_t, _, M_s_p_c_t, Psi_s_p_t = fast_predict_step(m_fn_z_s_tm1, Q_diag)

                kl_t = low_rank_kl_step_t(m_s_t, m_s_p_t, M_p_c_t, M_s_p_c_t,
                                          K_y[:, t], K_b[:, t],
                                          Psi_p_t, Psi_f_t, Psi_s_p_t, Psi_s_t, Q_diag)

            kl.append(kl_t)
            z_s.append(z_s_t)
            z_f.append(z_f_t)
            m_s.append(m_s_t)
            m_f.append(m_f_t)
            m_p.append(m_p_t)
            Psi_f.append(Psi_f_t)
            Psi_p.append(Psi_p_t)

        z_s = torch.stack(z_s, dim=2)
        stats['kl'] = torch.stack(kl, dim=1)
        stats['m_s'] = torch.stack(m_s, dim=1)
        stats['m_f'] = torch.stack(m_f, dim=1)
        stats['m_p'] = torch.stack(m_p, dim=1)
        stats['Psi_f'] = torch.stack(Psi_f, dim=1)
        stats['Psi_p'] = torch.stack(Psi_p, dim=1)

        return z_s, stats


# @torch.jit.script
def fast_J_p_bqp(M_p_c, Q_inv_diag, Psi_p, v):
    qp_1 = bip(Q_inv_diag[None, :] * v, v)

    Q_inv_M_p = (Q_inv_diag[None, :, None] * M_p_c)
    u = bmv(Psi_p.mT @ Q_inv_M_p.mT, v)
    qp_2 = bip(u, u)

    qp = qp_1 - qp_2
    return qp

def fast_tr_J_s_p_P_s(M_f_p_c, M_s_p_c, A, B, Psi_f, Psi_s_p, Psi_s, Q_diag):
    Q_inv_diag = 1 / Q_diag
    L = Q_inv_diag.shape[-1]
    Q_inv_sqrt_diag = torch.sqrt(Q_inv_diag)

    tr_1_sqrt = M_f_p_c * Q_inv_sqrt_diag[None, :, None]

    tr_2_sqrt = Q_inv_sqrt_diag[None, :, None] * fast_bmm_P_p(M_f_p_c, Q_diag, A @ Psi_f)
    tr_3_sqrt = Q_inv_sqrt_diag[None, :, None] * fast_bmm_P_f(A, Psi_f, M_f_p_c, Q_diag, B @ Psi_s)

    tr_4_term_1 = fast_bmm_P_s(Psi_f, Psi_s, B, A, M_f_p_c, Q_diag, (Q_inv_diag[None, :, None] * M_s_p_c) @ Psi_s_p)
    tr_4_term_2 = Psi_s_p.mT @ (Q_inv_diag[None, :, None] * M_s_p_c).mT

    tr_1 = torch.sum(tr_1_sqrt.pow(2), dim=[-1, -2])
    tr_2 = torch.sum(tr_2_sqrt.pow(2), dim=[-1, -2])
    tr_3 = torch.sum(tr_3_sqrt.pow(2), dim=[-1, -2])
    tr_4 = torch.sum(tr_4_term_1.mT * tr_4_term_2, dim=[-1, -2])

    tr = L + tr_1 - tr_2 - tr_3 - tr_4
    return tr

def log_det_kl_t(Psi_s_p, Psi_f, Psi_s, Psi_f_p):
    term_1 = -2 * torch.sum(torch.log(torch.diagonal(Psi_s_p, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    term_2 = -2 * torch.sum(torch.log(torch.diagonal(Psi_s, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    term_3 = -2 * torch.sum(torch.log(torch.diagonal(Psi_f, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    term_4 = -2 * torch.sum(torch.log(torch.diagonal(Psi_f_p, dim1=-2, dim2=-1) + 1e-8), dim=-1)

    logdet = term_1 + term_2 + term_3 - term_4

    return logdet


def low_rank_kl_step_t(m_s, m_s_p, M_f_p_c, M_s_p_c, A, B, Psi_f_p, Psi_f, Psi_s_p, Psi_s, Q_diag):
    Q_inv_diag = 1 / Q_diag
    L = m_s.shape[-1]

    tr = fast_tr_J_s_p_P_s(M_f_p_c, M_s_p_c, A, B, Psi_f, Psi_s_p, Psi_s, Q_diag)
    qp = fast_J_p_bqp(M_s_p_c, Q_inv_diag, Psi_s_p, m_s - m_s_p)
    log_det = log_det_kl_t(Psi_s_p, Psi_f, Psi_s, Psi_f_p)
    kl = 0.5 * (qp + tr + log_det - L)
    return kl

# @torch.jit.script
def low_rank_kl_step_0(m_s, m_0, Q_0_diag, Q_diag, A, B, Psi_f, Psi_s):
    L = m_s.shape[-1]
    delta_m = m_s - m_0
    Q_0_inv_diag = 1 / Q_0_diag
    Q_0_sqrt_diag = torch.sqrt(Q_0_diag)
    Q_0_sqrt_inv_diag = 1 / Q_0_sqrt_diag

    qp = bip(delta_m, Q_0_inv_diag * delta_m)

    log_det_1 = -2 * torch.sum(torch.log(torch.diagonal(Psi_s, dim1=-2, dim2=-1)), dim=-1)
    log_det_2 = -2 * torch.sum(torch.log(torch.diagonal(Psi_f, dim1=-2, dim2=-1)), dim=-1)
    log_det = log_det_1 + log_det_2

    tr_1_sqrt = (Q_0_sqrt_diag[None, :, None] * A) @ Psi_f
    tr_2_sqrt = Q_0_sqrt_inv_diag[None, :, None] * fast_bmm_P_f_0(A, Psi_f, Q_0_diag, B @ Psi_s)
    tr_1 = torch.sum(tr_1_sqrt.pow(2), dim=[-1, -2])
    tr_2 = torch.sum(tr_2_sqrt.pow(2), dim=[-1, -2])
    tr = L - tr_1 - tr_2

    kl = 0.5 * (qp + tr + log_det - L)

    return kl


# @torch.jit.script
def fast_bmv_P_p(M_c_p, Q_diag, v):
    u_1 = bmv(M_c_p, bmv(M_c_p.mT, v))
    u_2 = Q_diag * v
    u = u_1 + u_2
    return u


# @torch.jit.script
def fast_bmv_P_f(K, Psi_f, M_c_p, Q_diag, v):
    u_1 = fast_bmv_P_p(M_c_p, Q_diag, v)

    triple_bmv = bmv(K, bmv(Psi_f, bmv(Psi_f.mT, bmv(K.mT, u_1))))
    u_2 = fast_bmv_P_p(M_c_p, Q_diag, triple_bmv)
    u = u_1 - u_2
    return u


# @torch.jit.script
def fast_bmv_P_p_inv(Q_diag, M_c_p, Psi_p, v):
    Q_inv_diag = 1 / Q_diag

    u_1 = Q_inv_diag * v
    u_2 = Q_inv_diag * bmv(M_c_p, bmv(Psi_p, bmv(Psi_p.mT, bmv(M_c_p.mT, u_1))))
    u = u_1 - u_2
    return u


# @torch.jit.script
def fast_P_f_diagonal(K, Psi_f, M_c_p, Q_diag):
    L = K.shape[-2]
    e_basis = torch.eye(L, device=K.device).view(L, L)
    p = torch.stack([fast_bmv_P_f(K, Psi_f, M_c_p, Q_diag, e_basis[i])[..., i] for i in range(L)], dim=-1)
    return p


# @torch.jit.script
def fast_bmv_P_f_0(K, Psi_f, P_p_diag, v):
    u_1 = P_p_diag * v

    triple_bmv = bmv(K, bmv(Psi_f, bmv(Psi_f.mT, bmv(K.mT, u_1))))
    u_2 = P_p_diag * triple_bmv
    u = u_1 - u_2
    return u

# @torch.jit.script
def fast_P_f_0_diagonal(K, Psi_f, P_p_diag):
    L = K.shape[-2]
    e_basis = torch.eye(L, device=K.device).view(L, L)
    p = torch.stack([fast_bmv_P_f_0(K, Psi_f, P_p_diag, e_basis[i])[..., i] for i in range(L)], dim=-1)
    return p


# @torch.jit.script
def fast_bmm_P_p(M_c_p, Q_diag, V):
    U_1 = M_c_p @ (M_c_p.mT @ V)
    U_2 = Q_diag[None, :, None] * V
    U = U_1 + U_2
    return U


# @torch.jit.script
def fast_bmm_P_f_0(K_y, Psi_f, Q_0_diag, V):
    U_1 = Q_0_diag[None, :, None] * V

    triple_bmm = K_y @ (Psi_f @ (Psi_f.mT @ (K_y.mT @ U_1)))
    U_2 = Q_0_diag[None, :, None] * triple_bmm
    U = U_1 - U_2
    return U


# @torch.jit.script
def fast_bmm_P_f(K_y, Psi_f, M_c_p, Q_diag, V):
    U_1 = fast_bmm_P_p(M_c_p, Q_diag, V)

    W = K_y @ (Psi_f @ (Psi_f.mT @ (K_y.mT @ U_1)))
    U_2 = fast_bmm_P_p(M_c_p, Q_diag, W)
    U = U_1 - U_2
    return U


def fast_bmv_P_s(Psi_f, Psi_s, K_b, K_y, M_c_p, Q_diag, v):
    # TODO: optimize
    u_1 = fast_bmv_P_f(K_y, Psi_f, M_c_p, Q_diag, v)

    w = K_b @ (Psi_s @ (Psi_s.mT @ (K_b.mT @ u_1.unsqueeze(-1))))
    u_2 = fast_bmv_P_f(K_y, Psi_f, M_c_p, Q_diag, w.squeeze(-1))
    u = u_1 - u_2
    return u

def fast_bmm_P_s(Psi_f, Psi_s, K_b, K_y, M_c_p, Q_diag, V):
    U_1 = fast_bmm_P_f(K_y, Psi_f, M_c_p, Q_diag, V)

    W = K_b @ (Psi_s @ (Psi_s.mT @ (K_b.mT @ U_1)))
    U_2 = fast_bmm_P_f(K_y, Psi_f, M_c_p, Q_diag, W)
    U = U_1 - U_2
    return U


def fast_bmv_P_s_0(Psi_f, Psi_s, K_b, K_y, Q_0_diag, v):
    u_1 = fast_bmv_P_f_0(K_y, Psi_f, Q_0_diag, v)

    w = K_b @ (Psi_s @ (Psi_s.mT @ (K_b.mT @ u_1.unsqueeze(-1))))
    u_2 = fast_bmv_P_f_0(K_y, Psi_f, Q_0_diag, w.squeeze(-1))
    u = u_1 - u_2
    return u



def fast_update_filtering_to_smoothing_stats_0(z_f, h_f, m_f, Psi_f, k_b, K_b, K_y, Q_0_diag):
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
    z_s = m_s + z_f_c - bmv(K_b, chol_bmv_solve(I_r_pl_triple_chol, v_1))

    return m_s, z_s, Psi_s


def fast_update_filtering_to_smoothing_stats_t(z_f, h_f, m_f, Psi_f, M_c_f_p, k_b, K_b, K_y, Q_diag):
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
    z_s = m_s + z_f_c - bmv(K_b, chol_bmv_solve(I_r_pl_triple_chol, v_1))

    return m_s, z_s, Psi_s



# @torch.jit.script
def fast_update_step(z_p_c, h_p, k, K, w_f, M_c_p, Q_diag):
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
    z = m + z_p_c - bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    return m, z, Psi, h


# @torch.jit.script
def fast_predict_step(m_theta_z_tm1, Q_diag):
    device = m_theta_z_tm1.device
    n_trials, n_latents, S = m_theta_z_tm1.shape

    sqrt_S_inv = math.sqrt(1 / S)
    Q_diag_sqrt = torch.sqrt(Q_diag)
    I_S = torch.eye(S, device=m_theta_z_tm1.device)

    w_p_1 = torch.randn([S, n_trials, S], device=device)
    w_p_2 = torch.randn([S, n_trials, n_latents], device=device)

    m_p = m_theta_z_tm1.mean(dim=-1)
    M_c = sqrt_S_inv * (m_theta_z_tm1 - m_p.unsqueeze(-1))

    M_c_mT_Q_inv = M_c.mT * (1 / Q_diag)
    # I_pl_MmTQinvM_chol = torch.linalg.cholesky(I_S + M_c_mT_Q_inv @ M_c)
    I_pl_MmTQinvM_chol, _ = torch.linalg.cholesky_ex(I_S + M_c_mT_Q_inv @ M_c)
    Psi_p = linalg_utils.triangular_inverse(I_pl_MmTQinvM_chol).mT
    h_p = fast_bmv_P_p_inv(Q_diag, M_c, Psi_p, m_p)

    z_p_c = bmv(M_c, w_p_1) + Q_diag_sqrt * w_p_2

    return z_p_c, m_p, h_p, M_c, Psi_p


# @torch.jit.script
def fast_filter_step_t(m_theta_z_tm1, k, K, Q_diag, t_mask):
    device = m_theta_z_tm1.device
    n_trials, n_latents, rank = K.shape
    n_samples = m_theta_z_tm1.shape[-1]
    batch_sz = [n_trials]

    w_f = torch.randn([n_samples] + batch_sz + [rank], device=device)
    z_p_c, m_p, h_p, M_c_p, Psi_p = fast_predict_step(m_theta_z_tm1, Q_diag)

    if not t_mask:
        m_f, z_f, Psi_f, h_f = fast_update_step(z_p_c, h_p, k, K, w_f, M_c_p, Q_diag)
    else:
        h_f = h_p
        m_f = m_p
        z_f = m_p + z_p_c
        Psi_f = torch.ones((n_trials, rank, rank), device=device) * torch.eye(rank, device=device)

    return z_f, m_f, m_p, M_c_p, Psi_f, Psi_p, h_f


# @torch.jit.script
def fast_update_step_0(z_p_c, h_p, k, K, w_f, P_p_diag):
    n_trials, n_latents, rank = K.shape
    I_r = torch.eye(rank).to(z_p_c.device)

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
    z = m + z_p_c - bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))

    return m, z, Psi, h


# @torch.jit.script
def fast_filter_step_0(m_0: torch.Tensor, k: torch.Tensor, K: torch.Tensor, P_p_diag: torch.Tensor, n_samples: int):
    n_trials, n_latents, rank = K.shape
    batch_sz = [n_trials]
    w_p = torch.randn([n_samples] + batch_sz + [n_latents]).to(m_0.device)

    z_p_c = torch.sqrt(P_p_diag) * w_p
    J_p_diag = 1 / P_p_diag
    m_p = m_0 * torch.ones(batch_sz + [n_latents]).to(m_0.device)
    h_p = J_p_diag * m_p

    w_f = torch.randn([n_samples] + batch_sz + [rank]).to(m_0.device)
    m_f, z_f, Psi_f, h_f = fast_update_step_0(z_p_c, h_p, k, K, w_f,  P_p_diag)

    return z_f, m_f, m_p, Psi_f, h_f
