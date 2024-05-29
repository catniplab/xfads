import os
os.environ["OMP_NUM_THREADS"] = "8"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "8"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "8"  # export NUMEXPR_NUM_THREADS=6

import math
import torch
import torch.nn as nn
import xfads.utils as utils
import xfads.prob_utils as prob_utils
import pytorch_lightning as lightning
import matplotlib.pyplot as plt

from hydra import compose, initialize
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from xfads.ssm_modules.likelihoods import PoissonLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM, LightningDMFCRSG
from xfads.smoothers.nonlinear_smoother import NonlinearFilter, LowRankNonlinearStateSpaceModel
# from dev.smoothers.nonlinear_smoother_causal_debug import NonlinearFilter, LowRankNonlinearStateSpaceModel


def main():
    torch.cuda.empty_cache()
    initialize(version_base=None, config_path="", job_name="dmfc_rsg")
    cfg = compose(config_name="config")

    # seeds = [1234, 1235, 1236]
    # seeds = [1235, 1236]
    seeds = [1239]
    n_bins_bhv = 140

    """config"""
    for seed in seeds:
        cfg.seed = seed

        lightning.seed_everything(cfg.seed, workers=True)
        torch.set_default_dtype(torch.float32)

        """data"""
        data_path = 'data/old_data/data_{split}_{bin_sz_ms}ms.pt'
        train_data = torch.load(data_path.format(split='train', bin_sz_ms=cfg.bin_sz_ms))
        val_data = torch.load(data_path.format(split='valid', bin_sz_ms=cfg.bin_sz_ms))
        test_data = torch.load(data_path.format(split='test', bin_sz_ms=cfg.bin_sz_ms))

        y_valid_obs = val_data['y_obs'].type(torch.float32).to(cfg.data_device)
        y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)
        y_test_obs = test_data['y_obs'].type(torch.float32).to(cfg.data_device)
        n_trials, n_time_bins, n_neurons_obs = y_train_obs.shape
        n_time_bins_enc = train_data['n_time_bins_enc']

        y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, )
        y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, )
        y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, )
        train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(y_test_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)

        """likelihood pdf"""
        H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
        readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))
        readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz)
        likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

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
        local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                          device=cfg.device, dropout=cfg.p_local_dropout)
        nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

        """sequence vae"""
        ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                              local_encoder, nl_filter, device=cfg.device)

        """lightning"""
        # seq_vae = LightningDMFCRSG(ssm, cfg, n_time_bins_enc, n_bins_bhv)
        seq_vae = LightningDMFCRSG.load_from_checkpoint('ckpts/smoother/acausal/epoch=997_valid_loss=3288.73_valid_bps_enc=0.61_valid_bps_bhv=0.12.ckpt',
                                                        ssm=ssm, cfg=cfg, n_time_bins_enc=n_time_bins_enc, n_time_bins_bhv=n_bins_bhv, strict=False)
        csv_logger = CSVLogger('logs/smoother/acausal/', name=f'sd_{cfg.seed}_r_y_{cfg.rank_local}_r_b_{cfg.rank_backward}', version='smoother_acausal')
        ckpt_callback = ModelCheckpoint(save_top_k=3, monitor='valid_bps_enc', mode='max', dirpath='ckpts/smoother/acausal/', save_last=True,
                                        filename='{epoch:0}_{valid_loss:0.2f}_{valid_bps_enc:0.2f}_{valid_bps_bhv:0.2f}')

        trainer = lightning.Trainer(max_epochs=cfg.n_epochs,
                                    gradient_clip_val=1.0,
                                    default_root_dir='lightning/',
                                    callbacks=[ckpt_callback],
                                    logger=csv_logger,
                                    strategy='ddp',
                                    accelerator='gpu',
                                    )

        trainer.fit(model=seq_vae, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        torch.save(ckpt_callback.best_model_path, 'ckpts/smoother/acausal/best_model_path.pt')
        trainer.test(dataloaders=test_dataloader, ckpt_path='last')


if __name__ == '__main__':
    main()
