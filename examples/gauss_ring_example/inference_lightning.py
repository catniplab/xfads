import torch
import xfads.utils as utils
import pytorch_lightning as lightning
from hydra import compose, initialize
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL, LowRankNonlinearStateSpaceModel

import shutil

def main():
    torch.cuda.empty_cache()

    # config
    initialize(version_base=None, config_path="", job_name="gauss_ring")
    cfg = compose(config_name="config")
    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    # load data
    data = torch.load('gauss_ring_data.pt')
    y_obs = data['y_obs']
    z_true = data['z_true']

    n_trials, n_time_bins, n_neurons = y_obs.shape

    y_train, y_valid = y_obs[:n_trials//2], y_obs[n_trials//2:]
    y_train_dataset = torch.utils.data.TensorDataset(y_train,)
    y_valid_dataset = torch.utils.data.TensorDataset(y_valid,)
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(y_valid_dataset, batch_size=cfg.batch_sz, shuffle=True)

    # likelihood pdf
    # Use a linear mapping as the readout function, initialized to identity
    readout_fn = torch.nn.Linear(cfg.n_latents, n_neurons, bias=False, device=cfg.device)
    with torch.no_grad():
        # Set weights to identity (or as close as possible for rectangular)
        readout_fn.weight.data.zero_()
        for i in range(min(cfg.n_latents, n_neurons)):
            readout_fn.weight.data[i, i] = 1.0
    readout_fn.requires_grad_(False)
    R_diag = torch.ones(n_neurons, device=cfg.device) * 0.1
    likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device)

    # dynamics module
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    Q_diag = torch.ones(cfg.n_latents, device=cfg.device) * 1e-2
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

    # initial condition
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = torch.ones(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

    # local/backward encoder
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)
    nl_filter = NonlinearFilterSmallL(dynamics_mod, initial_condition_pdf, device=cfg.device)

    # sequence vae
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)

    # lightning
    seq_vae = LightningNonlinearSSM(ssm, cfg)
    csv_logger = CSVLogger('logs/', name=f'gauss_ring', version='noncausal')
    ckpt_callback = ModelCheckpoint(save_top_k=3, monitor='valid_loss', mode='min', dirpath='ckpts/',
                                    filename='{epoch:0}_{valid_loss}')

    trainer = lightning.Trainer(max_epochs=cfg.n_epochs,
                                gradient_clip_val=1.0,
                                default_root_dir='lightning/',
                                callbacks=[ckpt_callback],
                                logger=csv_logger
                                )

    trainer.fit(model=seq_vae, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    if ckpt_callback.best_model_path is not None and ckpt_callback.best_model_path != '':
        shutil.copy(ckpt_callback.best_model_path, 'ckpts/best_model_path.pt')

if __name__ == '__main__':
    main() 