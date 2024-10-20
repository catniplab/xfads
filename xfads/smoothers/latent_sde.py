import torch
from torch import nn
from functools import partial
from torch.distributions import Normal, Poisson

import torchsde


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class AddConst(nn.Module):
    def __init__(self, const):
        super(AddConst, self).__init__()
        self.const = const

    def forward(self, x):
        return x + self.const



class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, bin_sz, likelihood_pdf, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()
        self.bin_sz = bin_sz

        # Encoder.
        self.likelihood_pdf = likelihood_pdf
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid(),
                    AddConst(1e-2)
                )
                for _ in range(latent_size)
            ]
        )
        # self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs_pre, n_samples=1, kl_mult=1., noise_std=None, adjoint=False, method="euler", **kwargs):
        xs = xs_pre.transpose(1, 0)
        ts = self.bin_sz * torch.arange(xs.shape[0])

        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn([n_samples]+list(qz0_mean.shape), device=xs_pre.device)
        # z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=self.bin_sz, logqp=True, method=method)
        else:
            sde_int_partial = partial(torchsde.sdeint, self, ts=ts, dt=self.bin_sz, logqp=True, method=method)
            zs, log_ratio = torch.vmap(sde_int_partial, randomness='different')(z0)
            # zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=self.bin_sz, logqp=True, method=method)

        # _xs = self.projector(zs)
        _xs = self.likelihood_pdf.readout_fn(zs)
        # xs_dist = Normal(loc=_xs, scale=noise_std)
        # xs_dist = Poisson(rate=self.bin_sz * torch.exp(_xs))
        log_pxs = self.likelihood_pdf.get_ell(xs, zs).sum(dim=1).mean(dim=(0, 1))
        # log_pxs2 = xs_dist.log_prob(xs).sum(dim=(1, 3)).mean(dim=(0, 1))

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=1).mean(dim=(0, 1))


        stats = {}
        stats['m'] = zs.transpose(2, 1).mean(dim=0)
        stats['P_diag'] = zs.transpose(2, 1).var(dim=0)
        loss = -log_pxs + kl_mult * (logqp0 + logqp_path)
        return loss, zs.transpose(2, 1), stats
        # return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.likelihood_pdf.readout_fn(zs)
        # _xs = self.projector(zs)
        xs_dist = Poisson(rate=self.bin_sz * torch.exp(_xs))
        xs = xs_dist.sample()
        return xs, zs

    def predict_forward(self,
                        z_tm1: torch.Tensor,
                        n_bins: int):

        device = z_tm1.device
        n_samples, batch_size, latent_size = z_tm1.shape
        ts = torch.arange(n_bins) * self.bin_sz

        bm = torchsde.BrownianInterval(
            t0=0, t1=n_bins * self.bin_sz, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

        sde_int_partial = partial(torchsde.sdeint, self, ts=ts, dt=self.bin_sz, names={'drift': 'h'})
        zs = torch.vmap(sde_int_partial, randomness='different')(z_tm1)
        # zs = torchsde.sdeint(self, z_tm1.squeeze(0), ts, names={'drift': 'h'}, dt=self.bin_sz, bm=bm)
        return zs.transpose(2, 1)
