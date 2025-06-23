import math
import torch
from torch import nn
import torch.nn.functional as Fn
from .. import linalg_utils

from ..utils import pad_mask
from ..linalg_utils import bmv, bip, bop, chol_bmv_solve


class LowRankNonlinearStateSpaceModel(nn.Module):
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
    def forward(self,
                y,
                n_samples: int,
                p_mask_y_in: float=0.0,
                p_mask_apb: float = 0.0,
                p_mask_a: float = 0.0,
                p_mask_b: float = 0.0,
                get_P_s: bool = False):

        z_s, stats = self.fast_smooth_1_to_T(y, n_samples, p_mask_apb=p_mask_apb, p_mask_y_in=p_mask_y_in,
                                             p_mask_a=p_mask_a, p_mask_b=p_mask_b, get_kl=True, get_P_s=get_P_s)

        ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)
        loss = stats["kl"] - ell
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
            (1 - p_mask_y_in) * torch.ones((n_trials, n_time_bins, n_neurons))
        )
        t_mask_a = torch.bernoulli((1 - p_mask_a) * torch.ones((n_trials, n_time_bins)))

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        z_s, stats = self.nl_filter(k_y, K_y, n_samples, get_kl=get_kl, get_v=get_v)
        stats["t_mask_y_in"] = t_mask_y_in

        return z_s, stats

    def fast_smooth_1_to_T(self,
                           y,
                           n_samples: int,
                           p_mask_a: float=0.0,
                           p_mask_apb: float=0.0,
                           p_mask_y_in: float=0.0,
                           p_mask_b: float=0.0,
                           get_kl: bool=False,
                           get_v: bool=False,
                           get_P_s: bool=False):

        device = y.device
        n_trials, n_time_bins, n_neurons = y.shape

        t_mask_a = torch.bernoulli(
            (1 - p_mask_a) * torch.ones((n_trials, n_time_bins), device=y.device)
        )
        t_mask_b = torch.bernoulli(
            (1 - p_mask_b) * torch.ones((n_trials, n_time_bins), device=y.device)
        )
        t_mask_apb = torch.bernoulli(
            (1 - p_mask_apb) * torch.ones((n_trials, n_time_bins), device=y.device)
        )
        t_mask_y_in = torch.bernoulli(
            (1 - p_mask_y_in)
            * torch.ones((n_trials, n_time_bins, n_neurons), device=y.device)
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b
        K_b = t_mask_b[..., None, None] * K_b

        k = k_b + k_y
        K = torch.concat([K_b, K_y], dim=-1)
        k = t_mask_apb[..., None] * k
        K = t_mask_apb[..., None, None] * K

        z_s, stats = self.nl_filter(k, K, n_samples, get_kl=get_kl, get_v=get_v, get_P_s=get_P_s)
        stats['t_mask_y_in'] = t_mask_y_in

        return z_s, stats

    def predict_forward(self, z_tm1: torch.Tensor, n_bins: int):
        z_forward = []
        Q_sqrt = torch.sqrt(Fn.softplus(self.dynamics_mod.log_Q))

        for t in range(n_bins):
            if t == 0:
                z_t = self.dynamics_mod.mean_fn(z_tm1) + Q_sqrt * torch.randn_like(
                    z_tm1
                )
            else:
                z_t = self.dynamics_mod.mean_fn(
                    z_forward[t - 1]
                ) + Q_sqrt * torch.randn_like(z_forward[t - 1])

            z_forward.append(z_t)

        z_forward = torch.stack(z_forward, dim=2)
        return z_forward


class LrSSMcoBPSheldinEncoder(LowRankNonlinearStateSpaceModel):
    def __init__(
        self,
        dynamics_mod,
        likelihood_pdf,
        initial_c_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
        n_neurons_enc,
        n_neurons_obs,
        n_time_bins_enc,
    ):
        super().__init__()

        self.nl_filter = nl_filter
        self.dynamics_mod = dynamics_mod
        self.local_encoder = local_encoder
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf
        self.backward_encoder = backward_encoder

        self.n_neurons_enc = n_neurons_enc
        self.n_neurons_obs = n_neurons_obs
        self.n_time_bins_enc = n_time_bins_enc

    @torch.jit.export
    def forward(
        self,
        y_obs,
        n_samples: int,
        p_mask_apb: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        p_mask_y_in: float = 0.0,
        l2_C: float = 1e-1,
        use_cd=False,
    ):
        n_trials, _, n_neurons_obs = y_obs.shape

        z_enc, stats = self.fast_smooth_1_to_T(
            y_obs[..., : self.n_time_bins_enc, : self.n_neurons_enc],
            n_samples,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_apb=p_mask_apb,
            p_mask_b=p_mask_b,
            get_kl=True,
        )

        if use_cd:
            ell_enc = self.likelihood_pdf.get_ell(
                y_obs[:, : self.n_time_bins_enc], z_enc, reduce_neuron_dim=False
            ).mean(dim=0)
            grad_mask = pad_mask(stats["t_mask_y_in"], ell_enc, 0.0)

            ell_enc_no_grad = (ell_enc * grad_mask).detach()
            ell_grad = ell_enc * (1 - grad_mask)
            ell_enc = (ell_enc_no_grad + ell_grad).sum(dim=-1)
        else:
            ell_enc = self.likelihood_pdf.get_ell(
                y_obs[:, : self.n_time_bins_enc], z_enc
            ).mean(dim=0)

        C = self.likelihood_pdf.readout_fn[-1].weight
        loss_s = stats["kl"] - ell_enc
        loss_s = loss_s.sum(dim=-1).mean()
        loss_s += l2_C * C.pow(2).sum()
        stats["ell"] = ell_enc

        return loss_s, z_enc, stats

    @torch.jit.export
    def predict(
        self,
        y_enc,
        n_samples: int,
        p_mask_y_in: float = 0.0,
    ):
        z_s, stats = self.fast_smooth_1_to_T(y_enc, n_samples, get_kl=True)

        # expected log rate
        log_rate_hat = math.log(
            self.likelihood_pdf.delta
        ) + self.likelihood_pdf.readout_fn(stats["m_f"])
        stats["log_rate"] = log_rate_hat
        return z_s, stats


class LrSSMcoBPSallEncoder(LowRankNonlinearStateSpaceModel):
    def __init__(
        self,
        dynamics_mod,
        likelihood_pdf,
        initial_c_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
        n_neurons_enc,
        n_neurons_obs,
        n_time_bins_enc,
    ):
        super().__init__()

        self.nl_filter = nl_filter
        self.dynamics_mod = dynamics_mod
        self.local_encoder = local_encoder
        self.initial_c_pdf = initial_c_pdf
        self.likelihood_pdf = likelihood_pdf
        self.backward_encoder = backward_encoder

        self.n_neurons_enc = n_neurons_enc
        self.n_neurons_obs = n_neurons_obs
        self.n_time_bins_enc = n_time_bins_enc

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
            (1 - p_mask_y_in) * torch.ones((n_trials, 1, n_neurons))
        )

        y_in = t_mask_y_in * y / (1 - p_mask_y_in)

        k_y, K_y = self.local_encoder(y_in)
        k_y = t_mask_a[..., None] * k_y
        K_y = t_mask_a[..., None, None] * K_y

        k_b, K_b = self.backward_encoder(k_y, K_y)
        k_b = t_mask_b[..., None] * k_b
        K_b = t_mask_b[..., None, None] * K_b

        k = k_b + k_y
        K = torch.concat([K_b, K_y], dim=-1)
        k = t_mask_apb[..., None] * k
        K = t_mask_apb[..., None, None] * K

        z_s, stats = self.nl_filter(k, K, n_samples, get_kl=get_kl, get_v=get_v)
        stats["t_mask_y_in"] = t_mask_y_in

        return z_s, stats

    @torch.jit.export
    def forward(
        self,
        y_obs,
        n_samples: int,
        p_mask_apb: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        p_mask_y_in: float = 0.0,
        l2_C: float = 1e-1,
        use_cd=False,
    ):
        n_trials, _, n_neurons_obs = y_obs.shape

        z_enc, stats = self.fast_smooth_1_to_T(
            y_obs[..., : self.n_time_bins_enc, :],
            n_samples,
            p_mask_y_in=p_mask_y_in,
            p_mask_a=p_mask_a,
            p_mask_apb=p_mask_apb,
            p_mask_b=p_mask_b,
            get_kl=True,
        )

        ell_enc = self.likelihood_pdf.get_ell(
            y_obs[:, : self.n_time_bins_enc], z_enc
        ).mean(dim=0)
        C = self.likelihood_pdf.readout_fn[-1].weight
        loss_s = stats["kl"] - ell_enc
        loss_s = loss_s.sum(dim=-1).mean()
        loss_s += l2_C * C.pow(2).sum()
        stats["ell"] = ell_enc

        return loss_s, z_enc, stats

    @torch.jit.export
    def predict(self, y_enc, n_samples: int, p_mask_y_in: float = 0.0):
        n_neurons_heldout = self.n_neurons_obs - self.n_neurons_enc
        y_heldout = torch.zeros((y_enc.shape[0], y_enc.shape[1], n_neurons_heldout))
        y_input = torch.cat([y_enc / (1 - p_mask_y_in), y_heldout], dim=-1)
        z_s, stats = self.fast_smooth_1_to_T(y_input, n_samples, get_kl=True)

        # expected log rate
        log_rate_hat = math.log(
            self.likelihood_pdf.delta
        ) + self.likelihood_pdf.readout_fn(stats["m_f"])
        stats["log_rate"] = log_rate_hat
        return z_s, stats


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
        get_P_s: bool = False,
    ):
        # mask data, 0: data available, 1: data missing
        n_trials, n_time_bins, n_latents, rank = K.shape

        kl = []
        m_f = []
        z_f = []
        stats = {}

        if get_P_s:
            P_s = []

        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        Q_sqrt_diag = torch.sqrt(Q_diag)
        Q_inv_diag = 1 / Q_diag

        for t in range(n_time_bins):
            if t == 0:
                m_0 = self.initial_c_pdf.m_0
                P_p_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)
                z_f_t, m_f_t, m_p_t, Psi_f_t, P_p_diag = fast_filter_step_0(
                    m_0, k[:, 0], K[:, 0], P_p_diag, n_samples
                )

                if get_kl:
                    kl.append(
                        low_rank_kl_step_0(m_f_t, m_p_t, P_p_diag, K[:, 0], Psi_f_t)
                    )

                if get_P_s:
                    P_s.append(get_P_s_1(P_p_diag, K[:, 0]))

            else:
                m_fn_z_tm1 = self.dynamics_mod.mean_fn(z_f[t - 1]).movedim(0, -1)
                z_f_t, m_f_t, m_p_t, M_p_c_t, Psi_f_t, Psi_p_t = fast_filter_step_t(
                    m_fn_z_tm1, k[:, t], K[:, t], Q_diag, torch.tensor(False)
                )

                if get_kl:
                    kl.append(
                        low_rank_kl_step_t(
                            m_f_t,
                            m_p_t,
                            M_p_c_t,
                            K[:, t],
                            Psi_f_t,
                            Psi_p_t,
                            Q_inv_diag,
                            Q_sqrt_diag,
                        )
                    )

                if get_P_s:
                    P_s.append(get_P_s_t(Q_diag, M_p_c_t, K[:, t]))

            m_f.append(m_f_t)
            z_f.append(z_f_t)

        z_f = torch.stack(z_f, dim=2)
        stats["m_f"] = torch.stack(m_f, dim=1)

        if get_kl:
            stats["kl"] = torch.stack(kl, dim=1)

        if get_P_s:
            stats["P_s"] = torch.stack(P_s, dim=1)

        return z_f, stats


class NonlinearFilterSmallL(nn.Module):
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
        n_trials, n_time_bins, n_latents, rank = K.shape
        Q_diag = Fn.softplus(self.dynamics_mod.log_Q)
        t_mask = torch.rand(n_time_bins) < p_mask

        z_f = []
        m_p = []
        m_f = []
        P_p_chol = []
        P_f_chol = []
        stats = {}

        for t in range(n_time_bins):
            if t == 0:
                m_0 = self.initial_c_pdf.m_0
                P_0_diag = Fn.softplus(self.initial_c_pdf.log_Q_0)

                z_f_t, m_f_t, P_f_chol_t, P_p_chol_t = filter_step_0(
                    m_0, k[:, 0], K[:, 0], P_0_diag, n_samples
                )
                m_p.append(m_0 * torch.ones(n_trials, n_latents, device=m_0.device))
            else:
                m_fn_z_tm1 = self.dynamics_mod.mean_fn(z_f[t - 1]).movedim(0, -1)
                z_f_t, m_p_t, m_f_t, P_f_chol_t, P_p_chol_t = filter_step_t(
                    m_fn_z_tm1, k[:, t], K[:, t], Q_diag, t_mask[t]
                )
                m_p.append(m_p_t)

            z_f.append(z_f_t)
            m_f.append(m_f_t)
            P_f_chol.append(P_f_chol_t)
            P_p_chol.append(P_p_chol_t)

        z_f = torch.stack(z_f, dim=2)
        stats["m_f"] = torch.stack(m_f, dim=1)
        stats["m_p"] = torch.stack(m_p, dim=1)
        stats["P_f_chol"] = torch.stack(P_f_chol, dim=1)
        stats["P_p_chol"] = torch.stack(P_p_chol, dim=1)

        kl = full_rank_mvn_kl(
            stats["m_f"], stats["P_f_chol"], stats["m_p"], stats["P_p_chol"]
        )
        stats["kl"] = kl

        return z_f, stats


def get_P_s_t(Q_diag, M_p_c_t, K):
    # TODO: optimize order of operations
    P_p_t = M_p_c_t @ M_p_c_t.mT + torch.diag(Q_diag)
    I_pl_triple = torch.eye(K.shape[-1]) + K.mT @ P_p_t @ K
    Psi_t = linalg_utils.triangular_inverse(torch.linalg.cholesky(I_pl_triple)).mT
    P_s_t = P_p_t - P_p_t @ K @ Psi_t @ Psi_t.mT @ K.mT @ P_p_t

    return P_s_t


def get_P_s_1(Q_0_diag, K):
    # TODO: optimize order of operations
    P_p_t = torch.diag(Q_0_diag)
    I_pl_triple = torch.eye(K.shape[-1]) + K.mT @ P_p_t @ K
    Psi_t = linalg_utils.triangular_inverse(torch.linalg.cholesky(I_pl_triple)).mT
    P_s_t = P_p_t - P_p_t @ K @ Psi_t @ Psi_t.mT @ K.mT @ P_p_t
    return P_s_t


"""big L"""


def fast_J_p_bqp(M_p_c, Q_inv_diag, Psi_p, v):
    qp_1 = bip(Q_inv_diag[None, :] * v, v)

    Q_inv_M_p = Q_inv_diag[None, :, None] * M_p_c
    u = bmv(Psi_p.mT @ Q_inv_M_p.mT, v)
    qp_2 = bip(u, u)

    qp = qp_1 - qp_2
    return qp


# @torch.jit.script
def fast_tr_J_p_P_f(M_p_c, K, Psi_f, Q_sqrt_diag):
    L = Q_sqrt_diag.shape[-1]

    K_Psi = K @ Psi_f
    triple_1 = K_Psi.mT @ M_p_c
    tr_1 = torch.sum(triple_1.pow(2), dim=[-1, -2])

    triple_2 = Q_sqrt_diag[None, :, None] * K_Psi
    tr_2 = torch.sum(triple_2.pow(2), dim=[-1, -2])

    tr = L - tr_1 - tr_2
    return tr


# @torch.jit.script
def low_rank_kl_step_t(m_f, m_p, M_p_c, K, Psi_f, Psi_p, Q_inv_diag, Q_sqrt_diag):
    L = m_f.shape[-1]
    tr = fast_tr_J_p_P_f(M_p_c, K, Psi_f, Q_sqrt_diag)
    qp = fast_J_p_bqp(M_p_c, Q_inv_diag, Psi_p, m_f - m_p)
    logdet = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_f, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )
    kl = 0.5 * (qp + tr + logdet - L)
    return kl


# @torch.jit.script
def low_rank_kl_step_0(m_f, m_p, P_p_diag, K, Psi_f):
    #  tr(J_p @ P_f) = L - tr(P_p_sqrt.mT @ K @ Psi_f @ Psi_f.mT @ K.mT @ P_p_sqrt)
    L = m_f.shape[-1]
    delta_m = m_f - m_p
    P_p_inv_diag = 1 / P_p_diag
    P_p_sqrt_diag = torch.sqrt(P_p_diag)

    K_Psi = K @ Psi_f
    triple = P_p_sqrt_diag[None, :, None] * K_Psi
    tr = L - torch.sum(triple.pow(2), dim=[-1, -2])
    qp = bip(P_p_inv_diag[None, :] * delta_m, delta_m)
    logdet = -2 * torch.sum(
        torch.log(torch.diagonal(Psi_f, dim1=-2, dim2=-1) + 1e-8), dim=-1
    )

    kl = 0.5 * (qp + tr + logdet - L)
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
    e_basis = torch.eye(L).view(L, L)
    p = torch.stack(
        [fast_bmv_P_f(K, Psi_f, M_c_p, Q_diag, e_basis[i])[..., i] for i in range(L)],
        dim=-1,
    )
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
    e_basis = torch.eye(L).view(L, L)
    p = torch.stack(
        [fast_bmv_P_f_0(K, Psi_f, P_p_diag, e_basis[i])[..., i] for i in range(L)],
        dim=-1,
    )
    return p


# @torch.jit.script
def fast_update_step(z_p_c, h_p, k, K, w_f, M_c_p, Q_diag):
    n_trials, n_latents, rank = K.shape
    Q_diag_sqrt = torch.sqrt(Q_diag)
    I_r = torch.eye(rank)

    h = h_p + k

    K_mT_M_c = K.mT @ M_c_p
    K_mT_Q_sqrt = K.mT * Q_diag_sqrt[None, None, :]
    K_mT_P_K = K_mT_M_c @ K_mT_M_c.mT + K_mT_Q_sqrt @ K_mT_Q_sqrt.mT
    I_r_pl_triple = I_r + K_mT_P_K
    I_r_pl_triple_chol, _ = torch.linalg.cholesky_ex(I_r_pl_triple)
    Psi = linalg_utils.triangular_inverse(I_r_pl_triple_chol).mT

    m = fast_bmv_P_f(K, Psi, M_c_p, Q_diag, h)

    v_1 = bmv(K.mT, z_p_c) + w_f
    # z = m + z_p_c - bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    unscaled_update = bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    scaled_update = fast_bmv_P_p(M_c_p, Q_diag, unscaled_update)
    z = m + z_p_c - scaled_update
    return m, z, Psi


# @torch.jit.script
def fast_predict_step(m_theta_z_tm1, w_p_1, w_p_2, Q_diag):
    S = w_p_1.shape[-1]
    sqrt_S_inv = math.sqrt(1 / S)
    Q_diag_sqrt = torch.sqrt(Q_diag)
    I_S = torch.eye(S)

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
    n_trials, n_latents, rank = K.shape
    n_samples = m_theta_z_tm1.shape[-1]
    batch_sz = [n_trials]

    w_f = torch.randn([n_samples] + batch_sz + [rank])
    w_p_1 = torch.randn([n_samples] + batch_sz + [n_samples])
    w_p_2 = torch.randn([n_samples] + batch_sz + [n_latents])

    z_p_c, m_p, h_p, M_c_p, Psi_p = fast_predict_step(
        m_theta_z_tm1, w_p_1, w_p_2, Q_diag
    )

    if not t_mask:
        m_f, z_f, Psi_f = fast_update_step(z_p_c, h_p, k, K, w_f, M_c_p, Q_diag)
    else:
        m_f = m_p
        z_f = m_p + z_p_c
        Psi_f = torch.ones((n_trials, rank, rank)) * torch.eye(rank)

    return z_f, m_f, m_p, M_c_p, Psi_f, Psi_p


# @torch.jit.script
def fast_update_step_0(z_p_c, h_p, k, K, w_f, P_p_diag):
    n_trials, n_latents, rank = K.shape
    I_r = torch.eye(rank)

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
    # z = m + z_p_c - bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    unscaled_update = bmv(K, chol_bmv_solve(I_r_pl_triple_chol, v_1))
    scaled_update = P_p_diag * unscaled_update
    z = m + z_p_c - scaled_update

    return m, z, Psi


# @torch.jit.script
def fast_filter_step_0(
    m_0: torch.Tensor,
    k: torch.Tensor,
    K: torch.Tensor,
    P_p_diag: torch.Tensor,
    n_samples: int,
):
    n_trials, n_latents, rank = K.shape
    batch_sz = [n_trials]
    w_p = torch.randn([n_samples] + batch_sz + [n_latents])

    z_p_c = torch.sqrt(P_p_diag) * w_p
    J_p_diag = 1 / P_p_diag
    m_p = m_0 * torch.ones(batch_sz + [n_latents])
    h_p = J_p_diag * m_p

    w_f = torch.randn([n_samples] + batch_sz + [rank])
    m_f, z_f, Psi_f = fast_update_step_0(z_p_c, h_p, k, K, w_f, P_p_diag)

    return z_f, m_f, m_p, Psi_f, P_p_diag


"""small L"""


def full_rank_mvn_kl(m_f, P_f_chol, m_p, P_p_chol):
    tr = torch.einsum(
        "...ii -> ...", torch.cholesky_solve(P_f_chol @ P_f_chol.mT, P_p_chol)
    )
    logdet1 = 2 * torch.sum(
        torch.log(torch.diagonal(P_f_chol, dim1=-2, dim2=-1)), dim=-1
    )
    logdet2 = 2 * torch.sum(
        torch.log(torch.diagonal(P_p_chol, dim1=-2, dim2=-1)), dim=-1
    )
    qp = bip(m_f - m_p, chol_bmv_solve(P_p_chol, m_f - m_p))
    kl = 0.5 * (tr + qp + logdet2 - logdet1 - m_f.shape[-1])

    return kl


def predict_step_t(m_theta_z_tm1, Q_diag):
    M = -0.5 * (torch.diag(Q_diag) + bop(m_theta_z_tm1, m_theta_z_tm1))

    m_p = m_theta_z_tm1.mean(dim=0)
    M_p = M.mean(dim=0)
    P_p = -2 * M_p - bop(m_p, m_p)
    return m_p, P_p


def filter_step_t(m_theta_z_tm1, k, K, Q_diag, t_mask):
    n_trials, n_latents, rank = K.shape
    n_samples = m_theta_z_tm1.shape[-1]
    batch_sz = [n_trials]

    w_f = torch.randn([n_samples] + batch_sz + [n_latents], device=m_theta_z_tm1.device)
    m_p, P_p = predict_step_t(m_theta_z_tm1.movedim(-1, 0), Q_diag)
    P_p_chol = torch.linalg.cholesky(P_p)

    if not t_mask:
        h_p = chol_bmv_solve(P_p_chol, m_p)
        h_f = h_p + k

        J_p = torch.cholesky_inverse(P_p_chol)
        J_f = J_p + K @ K.mT
        J_f_chol = torch.linalg.cholesky(J_f)
        P_f_chol = linalg_utils.triangular_inverse(J_f_chol).mT
        m_f = chol_bmv_solve(J_f_chol, h_f)
    else:
        m_f = m_p
        P_f_chol = P_p_chol

    z_f = m_f + bmv(P_f_chol, w_f)

    return z_f, m_p, m_f, P_f_chol, P_p_chol


# @torch.jit.script
def filter_step_0(
    m_0: torch.Tensor,
    k: torch.Tensor,
    K: torch.Tensor,
    P_0_diag: torch.Tensor,
    n_samples: int,
):
    n_trials, n_latents, rank = K.shape
    batch_sz = [n_trials]

    J_0_diag = 1 / P_0_diag
    h_0 = J_0_diag * m_0
    J_f = torch.diag(J_0_diag) + K @ K.mT
    J_f_chol = torch.linalg.cholesky(J_f)
    P_f_chol = linalg_utils.triangular_inverse(J_f_chol).mT

    h_f = h_0 + k
    m_f = chol_bmv_solve(J_f_chol, h_f)

    P_p_chol = torch.diag(torch.sqrt(P_0_diag)) + torch.zeros_like(P_f_chol)
    w_f = torch.randn([n_samples] + batch_sz + [n_latents], device=m_f.device)
    z_f = m_f + bmv(P_f_chol, w_f)

    return z_f, m_f, P_f_chol, P_p_chol
