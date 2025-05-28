import torch
import torch.nn as nn
import xfads.utils as utils
import matplotlib.pyplot as plt
import xfads.prob_utils as prob_utils
import pytorch_lightning as lightning

from hydra import compose, initialize
from sklearn.linear_model import Ridge
from xfads.ssm_modules.likelihoods import PoissonLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningMonkeyReaching
from xfads.smoothers.nonlinear_smoother_causal import (
    NonlinearFilter,
    LowRankNonlinearStateSpaceModel,
)


def main():
    torch.cuda.empty_cache()
    initialize(version_base=None, config_path="", job_name="monkey_reaching")
    cfg = compose(config_name="config")
    cfg.data_device = "cpu"
    cfg.device = "cpu"
    n_bins_bhv = 10

    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    """data"""
    data_path = "data/data_{split}_{bin_sz_ms}ms.pt"
    train_data = torch.load(data_path.format(split="train", bin_sz_ms=cfg.bin_sz_ms))
    val_data = torch.load(data_path.format(split="test", bin_sz_ms=cfg.bin_sz_ms))
    test_data = torch.load(data_path.format(split="test", bin_sz_ms=cfg.bin_sz_ms))

    y_valid_obs = val_data["y_obs"].type(torch.float32).to(cfg.data_device)[:, :35]
    y_train_obs = train_data["y_obs"].type(torch.float32).to(cfg.data_device)[:, :35]
    y_test_obs = test_data["y_obs"].type(torch.float32).to(cfg.data_device)[:, :35]
    vel_valid = val_data["velocity"].type(torch.float32).to(cfg.data_device)
    vel_train = train_data["velocity"].type(torch.float32).to(cfg.data_device)
    vel_test = test_data["velocity"].type(torch.float32).to(cfg.data_device)
    n_trials, n_time_bins, n_neurons_obs = y_train_obs.shape
    n_time_bins_enc = train_data["n_time_bins_enc"]

    y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, vel_train)
    y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, vel_valid)
    y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, vel_test)
    train_dataloader = torch.utils.data.DataLoader(
        y_train_dataset, batch_size=cfg.batch_sz, shuffle=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        y_test_dataset, batch_size=y_valid_obs.shape[0], shuffle=False
    )

    """likelihood pdf"""
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))
    readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(
        train_dataloader, cfg.bin_sz
    )
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz)

    """dynamics module"""
    Q_diag = 1.0 * torch.ones(cfg.n_latents)
    dynamics_fn = utils.build_gru_dynamics_function(
        cfg.n_latents, cfg.n_hidden_dynamics
    )
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents)
    Q_0_diag = 1.0 * torch.ones(cfg.n_latents)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(
        cfg.n_latents,
        cfg.n_hidden_backward,
        cfg.n_latents,
        rank_local=cfg.rank_local,
        rank_backward=cfg.rank_backward,
    )
    local_encoder = LocalEncoderLRMvn(
        cfg.n_latents,
        n_neurons_obs,
        cfg.n_hidden_local,
        cfg.n_latents,
        rank=cfg.rank_local,
        dropout=cfg.p_local_dropout,
    )
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf)

    """sequence vae"""
    ssm = LowRankNonlinearStateSpaceModel(
        dynamics_mod,
        likelihood_pdf,
        initial_condition_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
    )

    """lightning"""
    best_model_path = "ckpts/smoother/causal_mask_0.0/epoch=827_valid_loss=1415.56_r2_valid_enc=0.89_r2_valid_bhv=0.00_valid_bps_enc=0.42.ckpt"
    seq_vae = LightningMonkeyReaching.load_from_checkpoint(
        best_model_path,
        ssm=ssm,
        cfg=cfg,
        n_time_bins_enc=n_time_bins_enc,
        n_time_bins_bhv=n_bins_bhv,
        strict=False,
    )
    seq_vae.ssm = seq_vae.ssm.to(cfg.device)
    seq_vae.ssm.eval()

    z_s_train = []
    z_s_valid = []
    z_f_valid = []
    z_p_valid = []

    for batch in train_dataloader:
        loss, z, stats = seq_vae.ssm(batch[0], cfg.n_samples)
        z_s_train.append(z)

    for batch in valid_dataloader:
        z_f, stats = seq_vae.ssm.fast_filter_1_to_T(batch[0], cfg.n_samples)
        loss, z, stats = seq_vae.ssm(batch[0], cfg.n_samples)
        z_p = seq_vae.ssm.predict_forward(z_f[:, :, 10], cfg.n_samples)
        z_p = torch.cat([z_f[:, :, :10], z_p], dim=2)
        z_f_valid.append(z_f)
        z_p_valid.append(z_p)
        z_s_valid.append(z)

    U, S, V = torch.svd(seq_vae.ssm.likelihood_pdf.readout_fn[-1].weight.data)
    V = S.unsqueeze(-1) * V

    z_s_train = torch.cat(z_s_train, dim=1)
    z_s_valid = torch.cat(z_s_valid, dim=1)
    z_f_valid = torch.cat(z_f_valid, dim=1)
    z_p_valid = torch.cat(z_p_valid, dim=1)

    rates = cfg.bin_sz * torch.exp(
        seq_vae.ssm.likelihood_pdf.readout_fn(z_s_train)
    ).mean(dim=0)
    rates_test = cfg.bin_sz * torch.exp(
        seq_vae.ssm.likelihood_pdf.readout_fn(z_s_valid)
    ).mean(dim=0)
    rates_filter = cfg.bin_sz * torch.exp(
        seq_vae.ssm.likelihood_pdf.readout_fn(z_f_valid)
    ).mean(dim=0)

    """velocity decoder"""
    with torch.no_grad():
        clf = Ridge(alpha=0.01)
        # fit to training data
        clf.fit(rates.reshape(-1, n_neurons_obs), vel_train.reshape(-1, 2))
        r2_test = clf.score(
            rates_test.reshape(-1, n_neurons_obs), vel_valid.reshape(-1, 2)
        )
        r2_filter = clf.score(
            rates_filter.reshape(-1, n_neurons_obs), vel_valid.reshape(-1, 2)
        )
        r2_k_step = []

        for k in range(30):
            z_prd_test = utils.propagate_latent_k_steps(
                z_f_valid[:, :, k], dynamics_mod, n_time_bins + 0 - (k + 1)
            )
            z_prd_test = torch.concat([z_f_valid[:, :, :k], z_prd_test], dim=2)

            # m_prd_test = z_prd_test.mean(dim=0)
            # m_prd_test = torch.concat([m_filter[:, :k], m_prd_test[:, k:]], dim=1)

            rates_prd_test = cfg.bin_sz * torch.exp(
                seq_vae.ssm.likelihood_pdf.readout_fn(z_prd_test)
            ).mean(dim=0)

            r2_prd = clf.score(
                rates_prd_test.reshape(-1, n_neurons_obs), vel_valid.reshape(-1, 2)
            )
            r2_k_step.append(r2_prd)

    plt.axvline(12, linestyle="--")
    plt.plot(r2_k_step)
    plt.axhline(r2_test, color="green", label="smoothed")
    plt.axhline(r2_filter, color="orange", label="filtered")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
