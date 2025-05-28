import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as Fn

from sklearn.linear_model import Ridge
from xfads.linalg_utils import bmv


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output


class DynamicsGRU(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim, use_layer_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru_cell = nn.GRUCell(0, hidden_dim)
        self.h_to_z = nn.Linear(hidden_dim, latent_dim)
        self.z_to_h = nn.Linear(latent_dim, hidden_dim)

        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.layer_norm = RMSNorm(latent_dim)

    def forward(self, z):
        h_in = self.z_to_h(z)
        h_in_shape = list(h_in.shape)[:-1]
        h_in = h_in.reshape((-1, self.hidden_dim))

        empty_vec = torch.empty((h_in.shape[0], 0), device=z.device)
        h_out = self.gru_cell(empty_vec, h_in)
        h_out = h_out.reshape(h_in_shape + [self.hidden_dim])
        residual = self.h_to_z(h_out)

        if self.use_layer_norm:
            out = self.layer_norm(z + residual)
        else:
            out = z + residual
        return out


class DynamicsQuadSaddle(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.delta = 0.05
        self.A_bd = torch.tensor([[-0.99, -0.4], [0.4, -0.99]])
        self.A_ul_lr = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        # self.A_ur_ll = self.A_ul_lr
        self.A_ur_ll = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
        # self.A_ur_ll = self.A_ul_lr

        self.z_0_ur = torch.tensor([1.0, 1.0])
        self.z_0_lr = torch.tensor([1.0, -1.0])
        self.z_0_ul = torch.tensor([-1.0, 1.0])
        self.z_0_ll = torch.tensor([-1.0, -1.0])
        # self.z_0_ul = 10 * torch.tensor([-1., 1.])
        # self.z_0_lr = 10 * torch.tensor([1., -1.])
        # self.z_0_ll = 10 * torch.tensor([-1., -1.])

    def forward(self, z):
        ub = 1.5
        y = torch.zeros_like(z)
        z_ur_dx = torch.where(
            (z[..., 0] >= 0)
            & (z[..., 1] > 0)
            & (torch.abs(z[..., 0]) < ub)
            & (torch.abs(z[..., 1]) < ub)
        )
        z_lr_dx = torch.where(
            (z[..., 0] > 0)
            & (z[..., 1] <= 0)
            & (torch.abs(z[..., 0]) < ub)
            & (torch.abs(z[..., 1]) < ub)
        )
        z_ul_dx = torch.where(
            (z[..., 0] <= 0)
            & (z[..., 1] > 0)
            & (torch.abs(z[..., 0]) < ub)
            & (torch.abs(z[..., 1]) < ub)
        )
        z_ll_dx = torch.where(
            (z[..., 0] < 0)
            & (z[..., 1] <= 0)
            & (torch.abs(z[..., 0]) < ub)
            & (torch.abs(z[..., 1]) < ub)
        )
        z_bd_dx = torch.where((torch.abs(z[..., 0]) > ub) | (torch.abs(z[..., 1]) > ub))

        if z.dim() == 3:
            y[z_ur_dx[0], z_ur_dx[1]] = bmv(
                self.A_ur_ll, z[z_ur_dx[0], z_ur_dx[1]] - self.z_0_ur
            )
            y[z_lr_dx[0], z_lr_dx[1]] = bmv(
                self.A_ul_lr, z[z_lr_dx[0], z_lr_dx[1]] - self.z_0_lr
            )
            y[z_ul_dx[0], z_ul_dx[1]] = bmv(
                self.A_ul_lr, z[z_ul_dx[0], z_ul_dx[1]] - self.z_0_ul
            )
            y[z_ll_dx[0], z_ll_dx[1]] = bmv(
                self.A_ur_ll, z[z_ll_dx[0], z_ll_dx[1]] - self.z_0_ll
            )
        elif z.dim() == 2:
            # eye = torch.eye(z.shape[-1])
            y[z_ur_dx[0]] = z[z_ur_dx[0]] + self.delta * bmv(
                self.A_ur_ll, z[z_ur_dx[0]] - self.z_0_ur
            )
            y[z_lr_dx[0]] = z[z_lr_dx[0]] + self.delta * bmv(
                self.A_ul_lr, z[z_lr_dx[0]] - self.z_0_lr
            )
            y[z_ul_dx[0]] = z[z_ul_dx[0]] + self.delta * bmv(
                self.A_ul_lr, z[z_ul_dx[0]] - self.z_0_ul
            )
            y[z_ll_dx[0]] = z[z_ll_dx[0]] + self.delta * bmv(
                self.A_ur_ll, z[z_ll_dx[0]] - self.z_0_ll
            )
            y[z_bd_dx[0]] = z[z_bd_dx[0]] + self.delta * bmv(self.A_bd, z[z_bd_dx[0]])

        # y = bmv(self.A_ul_lr, z)

        return y


class MGU(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.W_f = nn.Linear(latent_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.U_i = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.log_rho = nn.Parameter(torch.tensor(0.0))

        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.A.weight.data = 0.9 * make_2d_rotation_matrix(math.pi / 4)

    def forward(self, z_tm1):
        rho = torch.sigmoid(self.log_rho)

        U_i_z = self.U_i(z_tm1)
        f_t = Fn.sigmoid(self.W_f(z_tm1))
        h_tilde_t = Fn.tanh(self.W_h(f_t * U_i_z))
        h_t = (1 - f_t) * U_i_z + f_t * h_tilde_t
        z_t = rho * self.U_o(h_t) + (1 - rho) * self.A(z_tm1)

        return z_t


class ReadoutLatentMask(torch.nn.Module):
    def __init__(self, n_latents, n_latents_read):
        super().__init__()

        self.n_latents = n_latents
        self.n_latents_read = n_latents_read

    def forward(self, z):
        return z[..., : self.n_latents_read]

    def get_matrix_repr(self):
        H = torch.zeros((self.n_latents_read, self.n_latents))
        H[torch.arange(self.n_latents_read), torch.arange(self.n_latents_read)] = 1.0
        return H


class VdpDynamicsModel(nn.Module):
    def __init__(self, bin_sz=5e-3, mu=1.5, tau_1=0.1, tau_2=0.1):
        super().__init__()

        self.mu = mu
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.bin_sz = bin_sz

    def forward(self, z_t):
        tau_1_eff = self.bin_sz / self.tau_1
        tau_2_eff = self.bin_sz / self.tau_2

        z_tp1_d0 = z_t[..., 0] + tau_1_eff * z_t[..., 1]
        z_tp1_d1 = z_t[..., 1] + tau_2_eff * (
            self.mu * (1 - z_t[..., 0] ** 2) * z_t[..., 1] - z_t[..., 0]
        )
        z_tp1 = torch.concat([z_tp1_d0[..., None], z_tp1_d1[..., None]], dim=-1)

        return z_tp1


def build_gru_dynamics_function(
    dim_input, dim_hidden, d_type=torch.float32, use_layer_norm=False
):
    gru_dynamics = DynamicsGRU(dim_hidden, dim_input, use_layer_norm=use_layer_norm)
    return gru_dynamics


def softplus_inv(x):
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    else:
        return np.log(np.exp(x) - 1 + 1e-10)


def init_mlp_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        # torch.nn.init.orthogonal_(m.weight)
        # torch.nn.init.sparse_(m.weight, sparsity=0.5)
        try:
            m.bias.data.fill_(0.0)
        except:  # noqa: E722
            pass


def spike_resample_fn(y, resample_factor):
    arr_re = []

    for arr in y:
        np_arr_re = (
            np.nan_to_num(arr, copy=False)
            .reshape((arr.shape[0] // resample_factor, resample_factor, -1))
            .sum(axis=1)
        )
        arr_re.append(torch.tensor(np_arr_re).unsqueeze(0))

    return torch.concat(arr_re, dim=0)


def velocity_resample_fn(y, resample_factor):
    arr_re = []

    for arr in y:
        np_arr_re = (
            np.nan_to_num(arr, copy=False)
            .reshape((arr.shape[0] // resample_factor, resample_factor, -1))
            .mean(axis=1)
        )
        arr_re.append(torch.tensor(np_arr_re).unsqueeze(0))

    return torch.concat(arr_re, dim=0)


def make_2d_rotation_matrix(theta):
    A = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )

    return A


def pad_mask(mask, data, value):
    # source: https://github.com/arsedler9/lfads-torch

    """Adds padding to I/O masks for CD and SV in cases where
    reconstructed data is not the same shape as the input data.
    """
    t_forward = data.shape[1] - mask.shape[1]
    n_heldout = data.shape[2] - mask.shape[2]
    pad_shape = (0, n_heldout, 0, t_forward)
    return Fn.pad(mask, pad_shape, value=value)


def sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins):
    # input dim: (trial x time x n_inputs)
    # z_t = A @ z_{t-1} + v_t, v_t ~ N(0, Q)
    n_latents = Q_diag.shape[-1]
    z = torch.zeros((n_trials, n_time_bins, n_latents), device=Q_diag.device)

    for t in range(n_time_bins):
        if t == 0:
            z[:, 0] = m_0 + torch.sqrt(Q_0_diag) * torch.randn_like(z[:, 0])
        else:
            z[:, t] = mean_fn(z[:, t - 1]) + torch.sqrt(Q_diag) * torch.randn_like(
                z[:, t - 1]
            )

    return z


def evaluate_nlb_veloc_r2(
    cfg, ssm_nlb, train_dataloader, valid_dataloader, data_metadata
):
    n_neurons_enc = data_metadata["n_neurons_enc"]
    n_time_bins_enc = data_metadata["n_time_bins_enc"]

    clf = Ridge(alpha=0.01)
    # clf = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 0, 9)})
    train_veloc = []
    valid_veloc = []
    train_rates = []
    valid_rates = []

    with torch.no_grad():
        ssm_nlb.eval()

        for dx, batch in enumerate(train_dataloader):
            if dx == 0:
                n_neurons = batch[0].shape[-1]

            y_obs = batch[0]
            z_s_prd, stats_prd = ssm_nlb.predict(
                y_obs[..., :n_time_bins_enc, :n_neurons_enc], cfg.n_samples
            )
            train_rates.append(torch.exp(stats_prd["log_rate"][:, :n_time_bins_enc]))
            train_veloc.append(batch[1])

        for dx, batch in enumerate(valid_dataloader):
            y_obs = batch[0]
            z_s_prd, stats_prd = ssm_nlb.predict(
                y_obs[..., :n_time_bins_enc, :n_neurons_enc], cfg.n_samples
            )
            valid_rates.append(torch.exp(stats_prd["log_rate"][:, :n_time_bins_enc]))
            valid_veloc.append(batch[1])

    train_rates = torch.cat(train_rates, dim=0)
    valid_rates = torch.cat(valid_rates, dim=0)
    train_veloc = torch.cat(train_veloc, dim=0)
    valid_veloc = torch.cat(valid_veloc, dim=0)

    clf.fit(train_rates.reshape(-1, n_neurons), train_veloc.reshape(-1, 2))

    score = {}
    score["r2_train"] = clf.score(
        train_rates.reshape(-1, n_neurons), train_veloc.reshape(-1, 2)
    )
    score["r2_valid"] = clf.score(
        valid_rates.reshape(-1, n_neurons), valid_veloc.reshape(-1, 2)
    )

    return score


def get_updated_base_cfg(ray_cfg):
    base_cfg = ray_cfg["base_cfg"]
    cfg = copy.deepcopy(base_cfg)

    """set cfg values"""
    hyper_str = ""

    for k, v in ray_cfg.items():
        if k != "base_cfg" and k != "cwd":
            cfg[k] = ray_cfg[k]

            if isinstance(v, float):
                hyper_str += f"_{k}={v:.3f}"
            else:
                hyper_str += f"_{k}={v}"

    return cfg, hyper_str


class FanInLinear(nn.Linear):
    # source: https://github.com/arsedler9/lfads-torch
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


def propagate_latent_k_steps(z, dynamics_mod, k_steps):
    n_samples, n_trials, n_latents = z.shape
    Q = Fn.softplus(dynamics_mod.log_Q)
    mean_fn = dynamics_mod.mean_fn

    z_out = torch.zeros((n_samples, n_trials, k_steps + 1, n_latents), dtype=z.dtype)
    z_out[:, :, 0] = z

    for k in range(1, k_steps + 1):
        z_out[:, :, k] = mean_fn(z_out[:, :, k - 1]) + torch.sqrt(Q) * torch.randn_like(
            z_out[:, :, k - 1]
        )

    return z_out


class LowRankRegressor(nn.Module):
    def __init__(self, N, T, rank_n, rank_t):
        super().__init__()

        self.rank_n = rank_n
        self.rank_t = rank_t

        self.A = nn.Parameter(torch.rand((rank_t, T)))
        self.B = nn.Parameter(torch.rand((N, rank_n)))
        self.c = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return torch.sum(self.A @ x @ self.B, dim=[-2, -1]) + self.c
