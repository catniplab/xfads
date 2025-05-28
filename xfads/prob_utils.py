import math
import torch

from torch.nn.functional import poisson_nll_loss
from .linalg_utils import bip, bmv, chol_bmv_solve


def estimate_poisson_rate_bias(y, time_delta):
    if isinstance(y, torch.Tensor):
        bias_hat = torch.log(torch.mean(y, dim=[0, 1]) / time_delta + 1e-12)

    elif isinstance(y, torch.utils.data.DataLoader):
        n_batch = 0

        for dx, y_mb in enumerate(y):
            if dx == 0:
                full_batch_mean = torch.zeros(y_mb[0].shape[-1], device=y_mb[0].device)

            full_batch_mean += torch.mean(y_mb[0], dim=[0, 1])
            n_batch += 1

        full_batch_mean /= n_batch
        bias_hat = torch.log(full_batch_mean / time_delta + 1e-12)
    else:
        raise TypeError("pass in tensor or dataloader")

    return bias_hat


def bits_per_spike(preds, targets):
    # source: https://github.com/arsedler9/lfads-torch

    """
    Computes BPS for n_samples x n_timesteps x n_neurons arrays.
    Preds are logrates and targets are binned spike counts.
    """
    nll_model = poisson_nll_loss(preds, targets, full=True, reduction="sum")
    nll_null = poisson_nll_loss(
        torch.mean(targets, dim=(0, 1), keepdim=True),
        targets,
        log_input=False,
        full=True,
        reduction="sum",
    )
    return (nll_null - nll_model) / torch.nansum(targets) / math.log(2)


def rts_smoother(m_p, P_p, m_f, P_f, F):
    n_trials, n_time_bins, n_latents = m_p.shape

    m_s = [None] * n_time_bins
    P_s = [None] * n_time_bins

    m_s[-1] = m_f[:, -1]
    P_s[-1] = P_f[:, -1]

    for t in range(n_time_bins - 2, -1, -1):
        P_p_chol = torch.linalg.cholesky(P_p[:, t + 1])
        G = P_f[:, t] @ torch.cholesky_solve(F, P_p_chol).mT

        m_s[t] = m_f[:, t] + bmv(G, m_s[t + 1] - m_p[:, t + 1])
        P_s[t] = P_f[:, t] + G @ (P_s[t + 1] - P_p[:, t + 1]) @ G.mT

    m_s = torch.stack(m_s, dim=1)
    P_s = torch.stack(P_s, dim=1)

    return m_s, P_s


def kalman_information_filter(k, K, F, Q_diag, m_0, Q_0_diag):
    n_trials, n_time_bins, n_latents = k.shape

    Q_0 = torch.diag(Q_0_diag)
    Q = torch.diag(Q_diag)

    m_p = []
    m_f = []
    P_f = []
    P_p = []

    for t in range(n_time_bins):
        if t == 0:
            m_p.append(m_0 * torch.ones([n_trials, n_latents]))
            P_p.append(Q_0 * torch.ones([n_trials, n_latents, n_latents]))
        else:
            m_p.append(bmv(F, m_f[t - 1]))
            P_p.append(F @ P_f[t - 1] @ F.T + Q)

        P_p_chol = torch.linalg.cholesky(P_p[t])
        h_p = torch.cholesky_solve(m_p[t].unsqueeze(-1), P_p_chol).squeeze(-1)
        h_f_t = h_p + k[:, t]
        J_f_t = torch.cholesky_inverse(P_p_chol) + K[:, t]
        J_f_t_chol = torch.linalg.cholesky(J_f_t)
        m_f_t = torch.cholesky_solve(h_f_t.unsqueeze(-1), J_f_t_chol).squeeze(-1)
        P_f_t = torch.cholesky_inverse(J_f_t_chol)

        m_f.append(m_f_t)
        P_f.append(P_f_t)

    m_f = torch.stack(m_f, dim=1)
    P_f = torch.stack(P_f, dim=1)
    m_p = torch.stack(m_p, dim=1)
    P_p = torch.stack(P_p, dim=1)

    return m_f, P_f, m_p, P_p


def kl_diagonal_gaussian_canon(m_f, P_f_diag, m_p, P_p_diag):
    kl = (
        0.5 * torch.log(P_p_diag / P_f_diag)
        + 0.5 * (P_f_diag + (m_f - m_p) ** 2) / P_p_diag
        - 0.5
    )
    kl = kl.sum(dim=-1)
    return kl


def linear_gaussian_ell(y, C, b, R_diag, m, P):
    R_inv_diag = 1 / R_diag
    diff = y - bmv(C, m) - b

    qp = bip(diff, R_inv_diag * diff)
    logdet = torch.sum(torch.log(R_diag))
    tr = torch.einsum("...ii -> ...", (C.mT * R_inv_diag) @ C @ P)
    const = y.shape[-1] * math.log(2 * math.pi)

    ell = -0.5 * (qp + tr + logdet + const)

    return ell


def kl_dense_gaussian_full_rank(m_f, P_f_chol, m_p, P_p_chol):
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
