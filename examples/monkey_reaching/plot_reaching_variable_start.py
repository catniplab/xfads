import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as Fn
import xfads.prob_utils as prob_utils
import pytorch_lightning as lightning

from matplotlib import cm
from hydra import compose, initialize
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
from matplotlib.animation import FuncAnimation
from pyrecorder.converters.matplotlib import Matplotlib
import torch
import torch.nn as nn
import xfads.utils as utils
import matplotlib.pyplot as plt
import xfads.prob_utils as prob_utils
import xfads.plot_utils as plot_utils
import pytorch_lightning as lightning

from hydra import compose, initialize
from sklearn.linear_model import Ridge
from xfads.ssm_modules.likelihoods import PoissonLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM, LightningMonkeyReaching
from xfads.smoothers.nonlinear_smoother_causal import NonlinearFilter, LowRankNonlinearStateSpaceModel


def main():
    T = 35
    n_trials_plot = 8
    n_samples_mu_plt = 20
    predict_start_max = 8

    torch.cuda.empty_cache()
    initialize(version_base=None, config_path="", job_name="monkey_reaching")
    cfg = compose(config_name="config")
    cfg.data_device = 'cpu'
    cfg.device = 'cpu'
    n_bins_bhv=0

    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    """data"""
    data_path = 'data/data_{split}_{bin_sz_ms}ms.pt'
    train_data = torch.load(data_path.format(split='train', bin_sz_ms=cfg.bin_sz_ms))
    # val_data = torch.load(data_path.format(split='test', bin_sz_ms=cfg.bin_sz_ms))
    test_data = torch.load(data_path.format(split='test', bin_sz_ms=cfg.bin_sz_ms))

    # y_valid_obs = val_data['y_obs'].type(torch.float32).to(cfg.data_device)[:, :35]
    y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)[:, :35]
    y_test_obs = test_data['y_obs'].type(torch.float32).to(cfg.data_device)[:, :35]
    # vel_valid = val_data['velocity'].type(torch.float32).to(cfg.data_device)
    vel_train = train_data['velocity'].type(torch.float32).to(cfg.data_device)
    vel_test = test_data['velocity'].type(torch.float32).to(cfg.data_device)
    n_trials, n_time_bins, n_neurons_obs = y_train_obs.shape
    n_time_bins_enc = train_data['n_time_bins_enc']
    batch_sz_test = list(y_test_obs.shape)[:-1]
    n_trials_test = y_test_obs.shape[0]

    y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, vel_train)
    # y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, vel_valid)
    y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, vel_test)
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=False)
    # valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(y_test_dataset, batch_size=y_test_obs.shape[0], shuffle=False)

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
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents,
                                      rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

    """sequence vae"""
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)

    """lightning"""
    best_model_path = 'results/causal_model.ckpt'
    seq_vae = LightningMonkeyReaching.load_from_checkpoint(best_model_path, ssm=ssm, cfg=cfg,
                                                           n_time_bins_enc=n_time_bins_enc, n_time_bins_bhv=n_bins_bhv,
                                                           strict=False)
    seq_vae.ssm = seq_vae.ssm.to(cfg.device)
    seq_vae.ssm.eval()

    """inference"""
    z_s_train = []
    z_s_test = []
    z_f_test = []

    m_0 = seq_vae.ssm.initial_c_pdf.m_0
    Q_0 = Fn.softplus(seq_vae.ssm.initial_c_pdf.log_Q_0)
    z_ic = m_0 + Q_0.sqrt() * torch.randn([cfg.n_samples] + [batch_sz_test[0]] + [cfg.n_latents], device=cfg.device)
    z_ic_p = seq_vae.ssm.predict_forward(z_ic, T)

    for batch in train_dataloader:
        loss, z, stats = seq_vae.ssm(batch[0], cfg.n_samples)
        z_s_train.append(z)


    """training smoothed latents, test smoothed and filtered"""
    for batch in test_dataloader:
        z_f, stats = seq_vae.ssm.fast_filter_1_to_T(batch[0], cfg.n_samples)
        loss, z, stats = seq_vae.ssm(batch[0], cfg.n_samples)
        z_f_test.append(z_f)
        z_s_test.append(z)

    z_s_train = torch.cat(z_s_train, dim=1)
    z_s_test = torch.cat(z_s_test, dim=1)
    z_f_test = torch.cat(z_f_test, dim=1)

    """test predicted latents for all T horizons"""
    z_p_test = []

    for n_bins_bhv in range(predict_start_max):
        z_p_test_horizon_x = []

        for batch in test_dataloader:
            z_f, stats = seq_vae.ssm.fast_filter_1_to_T(batch[0], cfg.n_samples)
            z_p = seq_vae.ssm.predict_forward(z_f[:, :, n_bins_bhv], T-n_bins_bhv)
            z_p = torch.cat([z_f[:, :, :n_bins_bhv], z_p], dim=2)
            z_p_test_horizon_x.append(z_p)

        z_p_test.append(torch.cat(z_p_test_horizon_x, dim=1))

    rates_train_s = cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_s_train)).mean(dim=0)
    rates_test_s = cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_s_test)).mean(dim=0)
    rates_test_f = cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_f_test)).mean(dim=0)
    rates_test_p = [cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_p_test[k])).mean(dim=0) for k in range(predict_start_max)]
    rates_test_ic_p = cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_ic_p)).mean(dim=0)

    """plotting"""
    blues = cm.get_cmap("Blues", n_samples_mu_plt)
    grays = cm.get_cmap("Greys", n_samples_mu_plt)
    yellows = cm.get_cmap("YlOrBr", n_samples_mu_plt)

    """velocity decoder"""
    with torch.no_grad():
        clf = Ridge(alpha=0.01)
        # fit to training data
        clf.fit(rates_train_s.reshape(-1, n_neurons_obs), vel_train.reshape(-1, 2))
        r2 = clf.score(rates_train_s.reshape(-1, n_neurons_obs), vel_train.reshape(-1, 2))

        # transform test data
        r2_test_s = clf.score(rates_test_s.reshape(-1, n_neurons_obs), vel_test.reshape(-1, 2))
        r2_test_f = clf.score(rates_test_f.reshape(-1, n_neurons_obs), vel_test.reshape(-1, 2))
        r2_test_ic_p = clf.score(rates_test_ic_p.reshape(-1, n_neurons_obs), vel_test.reshape(-1, 2))
        r2_test_p = [clf.score(rates_test_p[k].reshape(-1, n_neurons_obs), vel_test.reshape(-1, 2)) for k in range(predict_start_max)]
        vel_hat_test_s = clf.predict(rates_test_s.reshape(-1, n_neurons_obs)).reshape(list(batch_sz_test) + [2])
        vel_hat_test_f = clf.predict(rates_test_f.reshape(-1, n_neurons_obs)).reshape(list(batch_sz_test) + [2])
        vel_hat_test_ic_p = clf.predict(rates_test_ic_p.reshape(-1, n_neurons_obs)).reshape(list(batch_sz_test) + [2])
        vel_hat_test_p = [clf.predict(rates_test_p[k].reshape(-1, n_neurons_obs)).reshape(list(batch_sz_test) + [2]) for k in range(predict_start_max)]

        vel_to_pos = lambda v: torch.cumsum(torch.tensor(v), dim=1)

        pos_test = vel_to_pos(vel_test)
        pos_test_hat_s = vel_to_pos(vel_hat_test_s)
        pos_test_hat_f = vel_to_pos(vel_hat_test_f)
        pos_test_hat_ic_p = vel_to_pos(vel_hat_test_ic_p)
        pos_test_hat_p = [vel_to_pos(vel_hat_test_p[k]) for k in range(predict_start_max)]

    trial_plt_dx = torch.randperm(n_trials_test)[:n_trials_plot]
    reach_angle = torch.atan2(pos_test[:, -1, 0], pos_test[:, -1, 1])
    reach_colors = plt.cm.hsv(reach_angle / (2 * np.pi) + 0.5)

    with torch.no_grad():
        fig, axs = plt.subplots(1, 4 + predict_start_max, figsize=(8*7, 3))

        plot_utils.plot_reaching(axs[0], pos_test[trial_plt_dx], reach_colors[trial_plt_dx])
        plot_utils.plot_reaching(axs[1], pos_test_hat_s[trial_plt_dx], reach_colors[trial_plt_dx])
        plot_utils.plot_reaching(axs[2], pos_test_hat_f[trial_plt_dx], reach_colors[trial_plt_dx])
        plot_utils.plot_reaching(axs[3], pos_test_hat_ic_p[trial_plt_dx], reach_colors[trial_plt_dx])
        [plot_utils.plot_reaching(axs[4 + k], pos_test_hat_p[k][trial_plt_dx], reach_colors[trial_plt_dx]) for k in range(predict_start_max)]

        axs[0].set_title('true')
        axs[1].set_title(f'smoothed, r2:{r2_test_s:.3f}')
        axs[2].set_title(f'filtered, r2:{r2_test_f:.3f}')
        axs[3].set_title(f'prior, r2:{r2_test_ic_p:.3f}')
        [axs[4 + k].set_title(f't_f={k}, r2:{r2_test_p[k]:.3f}') for k in range(predict_start_max)]
        fig.savefig('plots/filtered_vs_smoothed_vs_predicted_position.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()
