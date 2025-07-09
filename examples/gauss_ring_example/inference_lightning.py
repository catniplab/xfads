import torch
import pytorch_lightning as lightning
from hydra import compose, initialize
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
import shutil

from xfads.ssm_modules.dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL, LowRankNonlinearStateSpaceModel
import xfads.utils as utils


def main():
    torch.cuda.empty_cache()
    
    # config
    initialize(version_base=None, config_path="", job_name="enhanced_ring")
    cfg = compose(config_name="config")
    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)
    
    # Load data with external inputs injected into observations
    data = torch.load('gauss_ring_data.pt')
    y_obs = data['y_obs']  # [n_trials, n_time_bins, n_neurons] - observations with injected inputs
    z_true = data['z_true']  # [n_trials, n_time_bins, 2] - true latent trajectories
    u_inputs = data['u_inputs']  # [n_trials, n_time_bins, 2] - external inputs (for evaluation only)
    
    print("Data shapes:")
    print(f"  Observations: {y_obs.shape}")
    print(f"  True latents: {z_true.shape}")
    print(f"  External inputs: {u_inputs.shape} (for evaluation only)")
    
    n_trials, n_time_bins, n_neurons = y_obs.shape
    
    # Split data for training/validation (observations ONLY)
    split_idx = n_trials // 4
    y_valid, y_train = y_obs[:split_idx], y_obs[split_idx:]
    
    # Create datasets with ONLY observations (no inputs)
    train_dataset = torch.utils.data.TensorDataset(y_train)
    valid_dataset = torch.utils.data.TensorDataset(y_valid)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_sz, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_sz, shuffle=False)
    
    print(f"Training on {len(y_train)} trials, validating on {len(y_valid)} trials")
    
    # Create Gaussian likelihood with learnable readout
    readout_fn = utils.FanInLinear(cfg.n_latents, n_neurons, device=cfg.device)
    R_diag = torch.ones(n_neurons, device=cfg.device) * 0.1  # Observation noise
    likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device)
    
    # Dynamics module - learn ring + input effects as unified system
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    Q_diag = torch.ones(cfg.n_latents, device=cfg.device) * 1e-1
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)
    
    # Initial condition
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = torch.ones(cfg.n_latents, device=cfg.device) * 2.0
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)
    
    # Encoders
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)
    
    # Standard xFADS - NO input processing
    nl_filter = NonlinearFilterSmallL(dynamics_mod, initial_condition_pdf, device=cfg.device)
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, 
                                          backward_encoder, local_encoder, nl_filter, device=cfg.device)
    
    print("Created SSM with:")
    print(f"  Latent dimensions: {cfg.n_latents}")
    print(f"  Observation dimensions: {n_neurons}")
    print(f"  Dynamics hidden units: {cfg.n_hidden_dynamics}")
    print(f"  ⚠️  NO input processing - learning apparent dynamics")
    
    # Lightning wrapper - standard xFADS
    seq_vae = LightningNonlinearSSM(ssm, cfg)
    
    # Logging and checkpoints
    csv_logger = CSVLogger('logs/', name='enhanced_ring', version='no_inputs')
    ckpt_callback = ModelCheckpoint(
        save_top_k=3, 
        monitor='valid_loss', 
        mode='min', 
        dirpath='ckpts/',
        filename='enhanced_{epoch}_{valid_loss:.0f}'
    )
    
    # Trainer with gradient clipping for stability
    trainer = lightning.Trainer(
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.0,  # Standard clipping
        default_root_dir='lightning/',
        callbacks=[ckpt_callback],
        logger=csv_logger,
        check_val_every_n_epoch=5,
        log_every_n_steps=20
    )
    
    print("Starting training...")
    trainer.fit(model=seq_vae, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    
    # Save best model
    if ckpt_callback.best_model_path is not None and ckpt_callback.best_model_path != '':
        shutil.copy(ckpt_callback.best_model_path, 'ckpts/best_enhanced_model.pt')
        print(f"Best model saved from {ckpt_callback.best_model_path}")
        print(f"Training curves: logs/enhanced_ring/no_inputs/metrics.csv")
    
    print("✅ Training completed!")
    print("xFADS learned apparent dynamics (ring + input effects + noise)")

if __name__ == '__main__':
    main()