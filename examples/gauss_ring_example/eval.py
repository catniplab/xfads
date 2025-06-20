import torch
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
import pytorch_lightning as lightning
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL, LowRankNonlinearStateSpaceModel
import xfads.utils as utils


def plot_latents(z_inferred, z_true, inv_man=None, n_trials_plot=10):
    plt.figure(figsize=(8, 8))
    for i in range(n_trials_plot):
        plt.plot(z_inferred[i, :, 0], z_inferred[i, :, 1], color='red', alpha=0.5, label='Inferred' if i == 0 else None)
        plt.plot(z_true[i, :, 0], z_true[i, :, 1], color='black', alpha=0.5, label='True' if i == 0 else None)
    if inv_man is not None:
        plt.plot(inv_man[:, 0], inv_man[:, 1], color='blue', lw=2, label='Invariant Manifold')
    plt.xlabel('Latent 1')
    plt.ylabel('Latent 2')
    plt.title('Inferred vs True Latent Trajectories')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def main():
    # Load config
    initialize(version_base=None, config_path="", job_name="gauss_ring_eval")
    cfg = compose(config_name="config")
    torch.set_default_dtype(torch.float32)
    # Load data
    data = torch.load('gauss_ring_data.pt')
    y_obs = data['y_obs']
    z_true = data['z_true']
    inv_man = data['inv_man'] if 'inv_man' in data else None

    # --- reconstruct ssm as in training ---
    n_trials, n_time_bins, n_neurons = y_obs.shape

    # likelihood pdf
    readout_fn = torch.nn.Linear(cfg.n_latents, n_neurons, bias=False, device=cfg.device)
    with torch.no_grad():
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

    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)

    # --- load model ---
    seq_vae = LightningNonlinearSSM.load_from_checkpoint('ckpts/best_model_path.pt', ssm=ssm, cfg=cfg)
    # Infer latents
    with torch.no_grad():
        _, z_inferred, _ = seq_vae.ssm(y_obs, cfg.n_samples)
    # Plot
    plot_latents(z_inferred.mean(dim=0).cpu().numpy(), z_true.cpu().numpy(), inv_man.cpu().numpy() if inv_man is not None else None)

if __name__ == '__main__':
    main() 