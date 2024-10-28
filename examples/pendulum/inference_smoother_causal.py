import torch
import torch.nn as nn
import xfads.utils as utils
import pytorch_lightning as lightning

from hydra import compose, initialize
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.smoothers.lightning_trainers import LightningPendulum
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.nonlinear_smoother_causal import NonlinearFilter, LowRankNonlinearStateSpaceModel


def main():
    torch.cuda.empty_cache()
    initialize(version_base=None, config_path="", job_name="pendulum")
    cfg = compose(config_name="config")
    seeds = [1234, 1235, 1236, 1237]

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
        seq_vae = LightningPendulum(ssm, cfg, n_time_bins_enc)

        csv_logger = CSVLogger('logs/smoother/causal/', name=f'sd_{cfg.seed}_r_y_{cfg.rank_local}_r_b_{cfg.rank_backward}', version='smoother_causal')
        ckpt_callback = ModelCheckpoint(save_top_k=3, monitor='r2_valid_enc', mode='max', dirpath='ckpts/smoother/causal/',
                                        filename='{epoch:0}_{valid_mse:0.2f}_{r2_valid_prd:0.2f}_{r2_valid_enc:0.2f}_{valid_loss:0.2f}')

        trainer = lightning.Trainer(max_epochs=cfg.n_epochs,
                                    strategy='ddp_find_unused_parameters_true',
                                    gradient_clip_val=1.0,
                                    default_root_dir='lightning/',
                                    callbacks=[ckpt_callback],
                                    logger=csv_logger,
                                    check_val_every_n_epoch=cfg.check_val_every_n_epoch
                                    )

        trainer.fit(model=seq_vae, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        torch.save(ckpt_callback.best_model_path,f'ckpts/smoother/causal/sd_{cfg.seed}_best_model_path.pt')
        trainer.test(ckpt_path='best', dataloaders=test_dataloader)


if __name__ == '__main__':
    main()
