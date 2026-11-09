import torch
import torch.nn as nn
import torch.nn.functional as Fn

from xfads.decorators import *


@apply_memory_cleanup
class LocalEncoderLRMvn(nn.Module):
    def __init__(self, n_latents_read, input_size, hidden_size, n_latents, rank, device='cpu', dropout=0.0):
        super(LocalEncoderLRMvn, self).__init__()
        self.device = device

        self.rank = rank
        self.n_latents = n_latents
        self.n_latents_read = n_latents_read
        self.n_latents_unread = n_latents - n_latents_read

        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size, device=device),
                                 nn.SiLU(),
                                 nn.Dropout(dropout),
                                 # nn.Linear(hidden_size, (rank + 1) * n_latents, device=device)).to(device)
                                 nn.Linear(hidden_size, (rank + 1) * n_latents_read, device=device)).to(device)

    def forward(self, y):
        # missing entries of y are NaN: the MLP sees them as 0, and a bin with nothing observed gives no update
        obs = ~y.isnan()
        h_log_J = obs.any(-1, keepdim=True) * self.mlp(torch.where(obs, y, 0.))
        # h = h_log_J[..., :self.n_latents]
        h = h_log_J[..., :self.n_latents_read]
        # L_vec = h_log_J[..., self.n_latents:]
        L_vec = h_log_J[..., self.n_latents_read:]
        L = L_vec.view(y.shape[0], y.shape[1], self.n_latents_read, -1)
        # L = L_vec.view(y.shape[0], y.shape[1], self.n_latents, -1)

        h = torch.nn.functional.pad(h, (0, self.n_latents_unread), mode='constant', value=0.0)
        L = torch.nn.functional.pad(L, (0, 0, 0, self.n_latents_unread), mode='constant', value=0.0)

        # h[..., self.n_latents_read:] = h[..., self.n_latents_read:] * 0.0
        # L[..., self.n_latents_read:, :] = L[..., self.n_latents_read:, :] * 0.0

        return h, L


@apply_memory_cleanup
class BackwardEncoderLRMvn(nn.Module):
    def __init__(self, n_latents_read, hidden_size, n_latents, rank_local, rank_backward, device='cpu'):
        super(BackwardEncoderLRMvn, self).__init__()
        self.device = device
        self.n_latents_read = n_latents_read

        self.n_latents = n_latents
        self.rank_local = rank_local
        self.rank_backward = rank_backward

        self.rnn = torch.nn.GRU(input_size=n_latents * (rank_local + 1), hidden_size=hidden_size, batch_first=True,
                                bidirectional=False, device=device)

        self.projection = torch.nn.Linear(hidden_size, (rank_backward + 1) * n_latents, device=device)

    def forward(self, h_y, L_y, obs_bin=None):
        L_y_vec = L_y.view(h_y.shape[0], h_y.shape[1], -1)
        nat_y = torch.concat([h_y, L_y_vec], dim=-1)
        nat_y_flip = torch.flip(nat_y, dims=[1])

        # obs_bin (n_trials, n_time_bins) is False at bins with no observation. The GRU carries
        # its state through them, without a per-step loop: run it once on the observed bins
        # packed to the front (order kept), then read back at each bin the state after the last
        # observed bin read so far. With every bin observed the permutation and gathers are identities.
        if obs_bin is None:
            obs_bin = torch.ones(nat_y.shape[:2], dtype=torch.bool, device=nat_y.device)
        obs_flip = torch.flip(obs_bin, dims=[1])
        order = torch.argsort((~obs_flip).int(), dim=1, stable=True)
        w_packed, _ = self.rnn(nat_y_flip.gather(1, order[..., None].expand_as(nat_y_flip)))
        n_read = obs_flip.cumsum(dim=1)
        w_flip = w_packed.gather(1, (n_read - 1).clamp(min=0)[..., None].expand(-1, -1, w_packed.shape[-1]))

        w = torch.flip(w_flip, dims=[1])
        # no observation at or after a bin: no backward update; masked before the shift below,
        # so the returned update at t is zero when nothing is observed after t
        h_log_J = torch.flip(n_read > 0, dims=[1])[..., None] * self.projection(w)

        h = h_log_J[..., :self.n_latents]
        L_vec = h_log_J[..., self.n_latents:]
        L = L_vec.view(h_y.shape[0], h_y.shape[1], self.n_latents, self.rank_backward)

        # h[..., self.n_latents_read:] = 0.0
        # L[..., self.n_latents_read:, :] = 0.0

        h_out = torch.concat([h[:, 1:], h[:, -1:] * 0.], dim=1)
        L_out = torch.concat([L[:, 1:], L[:, -1:] * 0.], dim=1)

        return h_out, L_out


@apply_memory_cleanup
class LocalEncoderDiagonal(nn.Module):
    def __init__(self, n_latents_read, input_size, hidden_size, n_latents, rank, device='cpu', dropout=0.0):
        super(LocalEncoderDiagonal, self).__init__()
        self.device = device

        self.rank = rank
        self.n_latents = n_latents
        self.n_latents_read = n_latents_read

        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size, device=device),
                                 nn.SiLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_size, n_latents * 2, device=device)).to(device)

    def forward(self, y):
        h_log_J = self.mlp(y)
        h = h_log_J[..., :self.n_latents]
        L_vec = Fn.softplus(h_log_J[..., self.n_latents:])

        return h, L_vec


@apply_memory_cleanup
class BackwardEncoderDiagonal(nn.Module):
    def __init__(self, n_latents_read, hidden_size, n_latents, rank_local, rank_backward, device='cpu'):
        super(BackwardEncoderDiagonal, self).__init__()
        self.device = device
        self.n_latents_read = n_latents_read

        self.n_latents = n_latents
        self.rank_local = rank_local
        self.rank_backward = rank_backward

        self.rnn = torch.nn.GRU(input_size=n_latents * 2, hidden_size=hidden_size, batch_first=True,
                                bidirectional=False, device=device)

        self.projection = torch.nn.Linear(hidden_size, n_latents * 2, device=device)

    def forward(self, h_y, L_y):
        L_y_vec = L_y.view(h_y.shape[0], h_y.shape[1], -1)
        nat_y = torch.concat([h_y, L_y_vec], dim=-1)
        nat_y_flip = torch.flip(nat_y, dims=[1])
        w_flip, _ = self.rnn(nat_y_flip)

        w = torch.flip(w_flip, dims=[1])
        h_log_J = self.projection(w)

        h = h_log_J[..., :self.n_latents]
        L_vec = Fn.softplus(h_log_J[..., self.n_latents:])

        h_out = torch.concat([h[:, 1:], h[:, -1:] * 0.], dim=1)
        L_vec_out = torch.concat([L_vec[:, 1:], L_vec[:, -1:] * 0.], dim=1)

        return h_out, L_vec_out


@apply_memory_cleanup
class BackwardEncoderDVBF(nn.Module):
    def __init__(self, n_neurons, hidden_size, n_latents, device='cpu'):
        super(BackwardEncoderDVBF, self).__init__()
        self.device = device
        self.n_latents = n_latents

        self.rnn = torch.nn.GRU(input_size=n_neurons, hidden_size=hidden_size, batch_first=True,
                                bidirectional=False, device=device)

        self.projection = torch.nn.Linear(hidden_size, n_latents * 2, device=device)

    def forward(self, y):
        y_flip = torch.flip(y, dims=[1])
        w_flip, _ = self.rnn(y_flip)

        w = torch.flip(w_flip, dims=[1])
        m_log_P_diag = self.projection(w)

        m = m_log_P_diag[..., :self.n_latents]
        P_diag = Fn.softplus(m_log_P_diag[..., self.n_latents:])

        return m, P_diag


@apply_memory_cleanup
class LocalEncoderDVBF(nn.Module):
    def __init__(self, n_neurons, hidden_size, n_latents, device='cpu'):
        super(LocalEncoderDVBF, self).__init__()
        self.device = device
        self.n_latents = n_latents

        self.mlp = nn.Sequential(nn.Linear(n_neurons, hidden_size, device=device),
                                 nn.SiLU(),
                                 nn.Linear(hidden_size, 2 * n_latents, device=device))

    def forward(self, y):
        m_log_P_diag = self.mlp(y)

        m = m_log_P_diag[..., :self.n_latents]
        P_diag = Fn.softplus(m_log_P_diag[..., self.n_latents:])

        return m, P_diag


@apply_memory_cleanup
class BackwardEncoderDKF(nn.Module):
    def __init__(self, n_neurons, hidden_size, n_latents, device='cpu'):
        super(BackwardEncoderDKF, self).__init__()
        self.device = device
        self.n_latents = n_latents

        self.rnn = torch.nn.GRU(input_size=n_neurons, hidden_size=hidden_size, batch_first=True,
                                bidirectional=False, device=device)

    def forward(self, y):
        y_flip = torch.flip(y, dims=[1])
        w_flip, _ = self.rnn(y_flip)
        w = torch.flip(w_flip, dims=[1])
        return w


@apply_memory_cleanup
class LocalEncoderDKF(nn.Module):
    def __init__(self, n_hidden_local, n_hidden_backward, n_latents, device='cpu'):
        super(LocalEncoderDKF, self).__init__()
        self.device = device
        self.n_latents = n_latents

        self.projection = nn.Sequential(nn.Linear(n_hidden_backward + n_latents, n_hidden_local, device=device),
                                        nn.SiLU(),
                                        nn.Linear(n_hidden_local, 2 * n_latents, device=device))

    def forward(self, h, z):
        w = torch.cat([h, z], dim=-1)
        m_log_P_diag = self.projection(w)

        m = m_log_P_diag[..., :self.n_latents]
        P_diag = Fn.softplus(m_log_P_diag[..., self.n_latents:]).clip(min=1e-3)

        return m, P_diag


@apply_memory_cleanup
class InitEncoderDKF(nn.Module):
    def __init__(self, hidden_size, n_latents, device='cpu'):
        super(InitEncoderDKF, self).__init__()
        self.device = device
        self.n_latents = n_latents

        self.projection = nn.Linear(hidden_size, 2 * n_latents, device=device)

    def forward(self, h):
        m_log_P_diag = self.projection(h)

        m = m_log_P_diag[..., :self.n_latents]
        P_diag = Fn.softplus(m_log_P_diag[..., self.n_latents:])

        return m, P_diag


class MaskedInputEncoder(nn.Module):
    def __init__(self, n_inputs, n_latents, n_latents_read, device='cpu'):
        super(MaskedInputEncoder, self).__init__()

        self.device = device
        self.n_latents_read = n_latents_read
        self.n_latents_unread = n_latents - n_latents_read
        self.linear = nn.Linear(n_inputs, n_latents_read, bias=False, device=device)

    def forward(self, input):
        read_latent_update = self.linear(input)
        batch_sz = read_latent_update.shape[:-1]
        zero_pad = torch.zeros((*batch_sz, self.n_latents_unread), device=self.device)
        out = torch.cat([read_latent_update, zero_pad], dim=-1)

        return out
        
