import torch
import torch.nn as nn
import xfads.utils as utils
import matplotlib.pyplot as plt
import xfads.plot_utils as plot_utils
import pytorch_lightning as lightning

from hydra import compose, initialize
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM, LightningPendulum
from xfads.smoothers.nonlinear_smoother_causal import NonlinearFilter, LowRankNonlinearStateSpaceModel


def main():
    torch.cuda.empty_cache()
    initialize(version_base=None, config_path="", job_name="bouncing_ball")
    cfg = compose(config_name="config")
    seeds = [1236]
    # seeds = [1239]

    """config"""
    for seed in seeds:
        cfg.seed = seed

        lightning.seed_everything(cfg.seed, workers=True)
        torch.set_default_dtype(torch.float32)

        """data"""
        n_trials = 500
        n_time_bins = 100
        n_time_bins_enc = 50

        y_train = torch.load('data/split_data/y_train.pt').type(torch.float32)[:n_trials, :n_time_bins]
        z_train = torch.load('data/split_data/x_train_unscaled.pt').type(torch.float32)[:n_trials, :n_time_bins]
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 256))
        y_valid = torch.load('data/split_data/y_val.pt').type(torch.float32)[:n_trials, :n_time_bins]
        z_valid = torch.load('data/split_data/x_val_unscaled.pt').type(torch.float32)[:n_trials, :n_time_bins]
        y_valid = y_valid.reshape((y_valid.shape[0], y_valid.shape[1], 256))
        y_test = torch.load('data/split_data/y_test.pt').type(torch.float32)[:n_trials, :n_time_bins]
        z_test = torch.load('data/split_data/x_test_unscaled.pt').type(torch.float32)[:n_trials, :n_time_bins]
        y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 256))
        n_neurons = y_train.shape[-1]

        y_train_dataset = torch.utils.data.TensorDataset(y_train, z_train)
        y_valid_dataset = torch.utils.data.TensorDataset(y_valid, z_valid)
        y_test_dataset = torch.utils.data.TensorDataset(y_test, z_test)
        train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(y_valid_dataset, batch_size=cfg.batch_sz, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(y_test_dataset, batch_size=cfg.batch_sz, shuffle=True)

        """likelihood pdf"""
        R_diag = torch.ones(n_neurons, device=cfg.device)
        H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
        readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, 128), nn.SiLU(),
                                   nn.Linear(128, n_neurons), nn.Sigmoid())
        likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device)

        """dynamics module"""
        Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
        dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
        dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

        """initial condition"""
        m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
        Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
        initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

        """local/backward encoder"""
        backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                                rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                                device=cfg.device)
        local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                          device=cfg.device, dropout=cfg.p_local_dropout)
        nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

        """sequence vae"""
        ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                              local_encoder, nl_filter, device=cfg.device)

        """lightning"""
        seq_vae = LightningPendulum.load_from_checkpoint('ckpts/smoother/causal/epoch=1349_valid_mse=0.06_r2_valid_prd=0.62_r2_valid_enc=0.98_valid_loss=192.23.ckpt', ssm=ssm, cfg=cfg, n_time_bins_enc=n_time_bins_enc, strict=False)
        _, z_s, stats_s = seq_vae.ssm.to('cpu')(y_test[:, :n_time_bins_enc].to('cpu'), cfg.n_samples, get_P_s=True)
        z_p = seq_vae.ssm.to('cpu').predict_forward(z_s[:, :, -1], n_time_bins - n_time_bins_enc)

        y_s = seq_vae.ssm.to('cpu').likelihood_pdf.readout_fn(z_s)
        y_p = seq_vae.ssm.to('cpu').likelihood_pdf.readout_fn(z_p)

        y_hat = torch.cat([y_s, y_p], dim=2)
        z_hat = torch.cat([z_s, z_p], dim=2)

        i=27
        end_dx = 75
        start_dx = 40
        fig = plot_utils.plot_ball(y_test[i, start_dx:end_dx], y_hat[0, i, start_dx:end_dx],
                                   z_hat[0, i, start_dx:end_dx], label_dx=50-start_dx)
        fig.savefig('plots/pendulum_predictions_causal.pdf', bbox_inches='tight', transparent=True)
        fig.show()


        i=27
        end_dx = 50
        start_dx = 10
        fig = plot_utils.plot_ball_covariance(y_test[i, start_dx:end_dx],
                                              torch.diagonal(stats_s['P_s'][i, start_dx:end_dx], dim1=-2, dim2=-1),
                                              P=stats_s['P_s'][i, start_dx:end_dx],
                                              label_dx=50-start_dx)
        fig.show()



if __name__ == '__main__':
    main()
