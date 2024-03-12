import math
import torch

from torch.nn.functional import poisson_nll_loss
from dev.linalg_utils import bip, bop, bmv, bqp, chol_bmv_solve


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
        raise TypeError('pass in tensor or dataloader')

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
