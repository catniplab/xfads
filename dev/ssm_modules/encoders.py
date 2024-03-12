import torch
import torch.nn as nn

class LocalEncoderLRMvn(nn.Module):
    def __init__(self, n_latents_read, input_size, hidden_size, n_latents, rank, device='cpu', dropout=0.0):
        super(LocalEncoderLRMvn, self).__init__()
        self.device = device

        self.rank = rank
        self.n_latents = n_latents
        self.n_latents_read = n_latents_read

        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size, device=device),
                                 nn.SiLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_size, (rank + 1) * n_latents, device=device)).to(device)


    def forward(self, y):
        h_log_J = self.mlp(y)
        h = h_log_J[..., :self.n_latents]
        L_vec = h_log_J[..., self.n_latents:]
        L = L_vec.view(y.shape[0], y.shape[1], self.n_latents, -1)

        # h[..., self.n_latents_read:] = 0.0
        # L[..., self.n_latents_read:, :] = 0.0

        return h, L


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

    def forward(self, h_y, L_y):
        L_y_vec = L_y.view(h_y.shape[0], h_y.shape[1], -1)
        nat_y_hat = torch.concat([h_y, L_y_vec], dim=-1)
        w_flip, _ = self.rnn(nat_y_hat)

        w = torch.flip(w_flip, dims=[1])
        h_log_J = self.projection(w)

        h = h_log_J[..., :self.n_latents]
        L_vec = h_log_J[..., self.n_latents:]
        L = L_vec.view(h_y.shape[0], h_y.shape[1], self.n_latents, self.rank_backward)

        # h[..., self.n_latents_read:] = 0.0
        # L[..., self.n_latents_read:, :] = 0.0

        h_out = torch.concat([h[:, 1:], h[:, -1:] * 0.], dim=1)
        L_out = torch.concat([L[:, 1:], L[:, -1:] * 0.], dim=1)

        return h_out, L_out
