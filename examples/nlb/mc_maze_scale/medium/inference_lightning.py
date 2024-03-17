import torch
import dev.utils as utils
import matplotlib.pyplot as plt
import dev.prob_utils as prob_utils
import pytorch_lightning as lightning

from hydra import compose, initialize
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dev.ssm_modules.likelihoods import PoissonLikelihood
from dev.ssm_modules.dynamics import DenseGaussianDynamics
from dev.ssm_modules.dynamics import DenseGaussianInitialCondition
from dev.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from dev.smoothers.nonlinear_smoother import NonlinearFilter, LrSSMcoBPSheldinEncoder, LrSSMcoBPSallEncoder
from dev.smoothers.lightning_trainers import LightningNlbNonlinearSSM


def main():
    """config"""
    initialize(version_base=None, config_path="", job_name="medium")
    cfg = compose(config_name="config")
    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    """data"""
    data_path = 'data/data_{split}_{bin_sz_ms}ms.pt'
    train_data = torch.load(data_path.format(split='train', bin_sz_ms=cfg.bin_sz_ms))
    val_data = torch.load(data_path.format(split='valid', bin_sz_ms=cfg.bin_sz_ms))

    y_valid_obs = val_data['y_obs'].type(torch.float32).to(cfg.data_device)
    y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)
    vel_valid = val_data['velocity'].type(torch.float32).to(cfg.data_device)
    vel_train = train_data['velocity'].type(torch.float32).to(cfg.data_device)
    n_trials, n_time_bins, n_neurons_obs = y_train_obs.shape

    y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, vel_train)
    y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, vel_valid)
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)

    """likelihood pdf"""
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    C = utils.FanInLinear(cfg.n_latents_read, n_neurons_obs, device=cfg.device)
    readout_fn = torch.nn.Sequential(H, C)
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q = torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q, device=cfg.device)

    """initial condition"""
    Q0 = torch.ones(cfg.n_latents, device=cfg.device)
    m0 = torch.zeros(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m0, Q0, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents_read, cfg.n_hidden_backward, cfg.n_latents, cfg.rank_local,
                                            cfg.rank_backward, device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents_read, y_train_obs.shape[-1], cfg.n_hidden_local, cfg.n_latents,
                                      cfg.rank_local, device=cfg.device, dropout=cfg.p_local_dropout)
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

    """sequence vae"""
    ssm = LrSSMcoBPSallEncoder(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder, local_encoder,
                               nl_filter, train_data['n_neurons_enc'], y_train_obs.shape[-1],
                               train_data['n_time_bins_enc'], device=cfg.device)

    ssm.likelihood_pdf.readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(y_train_obs, cfg.bin_sz)

    """lightning"""
    seq_vae = LightningNlbNonlinearSSM(ssm, cfg)
    csv_logger = CSVLogger('logs/', name=f'r_y_{cfg.rank_local}_r_b_{cfg.rank_backward}', version='noncausal')
    ckpt_callback = ModelCheckpoint(save_top_k=3, monitor='val_bps_hld', mode='max', dirpath='models/',
                                    filename='{epoch:0}_{val_bps_hld:.3f}_{val_veloc_r2:.3f}', save_last=True)

    trainer = lightning.Trainer(max_epochs=cfg.n_epochs,
                                gradient_clip_val=1.0,
                                default_root_dir='lightning/',
                                callbacks=[ckpt_callback],
                                logger=csv_logger
                                )

    trainer.fit(model=seq_vae, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    best_model_path = ckpt_callback.best_model_path
    f = open('logs/best_ckpt_path.txt', 'w')
    f.write(best_model_path)
    f.close()


if __name__ == '__main__':
    main()
