import torch
import torch.nn as nn
import xfads.utils as utils
import matplotlib.pyplot as plt
import xfads.prob_utils as prob_utils
import xfads.plot_utils as plot_utils
import pytorch_lightning as lightning

from hydra import compose, initialize
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL, LowRankNonlinearStateSpaceModel



def main():
    torch.cuda.empty_cache()

    """config"""
    initialize(version_base=None, config_path="", job_name="lds")
    cfg = compose(config_name="config")
    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    """generate data -- 2d oscillator with decay"""
    n_trials = 500
    n_neurons = 100
    n_time_bins = 75

    mean_fn = utils.RingAttractorDynamics(bin_sz=1e-1, w=0.0)
    C = utils.FanInLinear(cfg.n_latents, n_neurons, device=cfg.device).requires_grad_(False)
    Q_diag = 5e-3 * torch.ones(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    R_diag = 1e-1 * torch.ones(n_neurons, device=cfg.device)
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)

    z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
    y = C(z) + torch.sqrt(R_diag) * torch.randn((n_trials, n_time_bins, n_neurons), device=cfg.device)
    y = y.detach()

    y_train, z_train = y[:2*n_trials//3], z[:2*n_trials//3]
    y_valid, z_valid = y[2*n_trials//3:], z[2*n_trials//3:]

    y_train_dataset = torch.utils.data.TensorDataset(y_train,)
    y_valid_dataset = torch.utils.data.TensorDataset(y_valid,)
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(y_valid_dataset, batch_size=cfg.batch_sz, shuffle=True)

    """likelihood pdf"""
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, C)
    likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device, fix_R=True)

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
    nl_filter = NonlinearFilterSmallL(dynamics_mod, initial_condition_pdf, device=cfg.device)

    """sequence vae"""
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)

    """lightning"""
    seq_vae = LightningNonlinearSSM.load_from_checkpoint('ckpts/example_model.ckpt', ssm=ssm, cfg=cfg)

    fig, axs = plt.subplots()
    plot_utils.plot_two_d_vector_field(seq_vae.ssm.dynamics_mod.mean_fn, axs)
    plt.show()


if __name__ == '__main__':
    main()
