import torch
import torch.nn as nn
import matplotlib as mpl
import xfads.utils as utils
import matplotlib.pyplot as plt
import xfads.plot_utils as plot_utils
import xfads.prob_utils as prob_utils
import pytorch_lightning as lightning

from matplotlib import cm
from hydra import compose, initialize
from scipy.ndimage import gaussian_filter1d
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from xfads.ssm_modules.likelihoods import PoissonLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM, LightningDMFCRSG
from xfads.smoothers.nonlinear_smoother import NonlinearFilter, LowRankNonlinearStateSpaceModel


def main():
    torch.cuda.empty_cache()
    initialize(version_base=None, config_path="", job_name="dmfc_rsg")
    cfg = compose(config_name="config")

    seeds = [1234]
    n_bins_bhv = 130

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
        ts_valid = val_data['ts'].type(torch.float32).to(cfg.data_device)
        ts_train = train_data['ts'].type(torch.float32).to(cfg.data_device)
        ts_test = test_data['ts'].type(torch.float32).to(cfg.data_device)
        task_id_valid = val_data['task_id'].type(torch.float32).to(cfg.data_device)
        task_id_train = train_data['task_id'].type(torch.float32).to(cfg.data_device)
        task_id_test = test_data['task_id'].type(torch.float32).to(cfg.data_device)
        n_trials, n_time_bins, n_neurons_obs = y_train_obs.shape
        n_time_bins_enc = train_data['n_time_bins_enc']

        y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, ts_train, task_id_train)
        y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, ts_valid, task_id_valid)
        y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, ts_test, task_id_test)
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
        seq_vae = LightningDMFCRSG.load_from_checkpoint('ckpts/smoother/acausal/model_path.ckpt',
                                                        ssm=ssm, cfg=cfg, n_time_bins_enc=n_time_bins_enc, n_time_bins_bhv=n_bins_bhv, strict=False)


        _, z_s, _ = seq_vae.ssm.to('cpu')(y_test_obs[:, :n_time_bins_enc].to('cpu'), cfg.n_samples)
        z_p = seq_vae.ssm.to('cpu').predict_forward(z_s[:, :, n_bins_bhv], n_time_bins_enc - n_bins_bhv)

        # smoothed and predicted log rates
        r_s = seq_vae.ssm.to('cpu').likelihood_pdf.readout_fn(z_s)
        r_p = seq_vae.ssm.to('cpu').likelihood_pdf.readout_fn(z_p)
        y_s = torch.poisson(cfg.bin_sz * torch.exp(r_s))
        y_p = torch.poisson(cfg.bin_sz * torch.exp(r_p))

        y_hat = torch.cat([y_s[:, :, :n_bins_bhv], y_p], dim=2)
        z_hat = torch.cat([z_s[:, :, :n_bins_bhv, :], z_p], dim=2)
        i = 41

        with torch.no_grad():
            fig, axs = plt.subplots(2, 1, figsize=(3, 3))
            plot_utils.plot_spikes(y_test_obs[i].cpu(), axs[0])
            plot_utils.plot_spikes(y_hat[0, i], axs[1])

            fig.suptitle(i)
            fig.savefig('plots/dmfc_spikes_predictions_acausal.svg', bbox_inches='tight', transparent=True)
            fig.show()

    task_id = 7
    colors = cm.get_cmap("spring", cfg.n_samples)

    for task_id in [4, 7, 8, 10, 11]:
        fig, axs = plt.subplots(10, 5)
        fig.suptitle(task_id)
        axs = axs.reshape(-1)

        for k in range(50):
            neuron_plot_id = k
            test_subset_dx = torch.where(task_id_test == task_id)[0]
            train_subset_dx = torch.where(task_id_train == task_id)[0]

            y_hat_subset = y_hat[:, test_subset_dx]
            y_subset = y_test_obs[test_subset_dx]
            # y_subset = y_train_obs[train_subset_dx]

            with torch.no_grad():
                # ground truth psth
                y_psth = torch.tensor(gaussian_filter1d(y_subset.mean(dim=0), sigma=10, axis=0))
                # psth sampled from generative model
                y_hat_psth = torch.tensor(gaussian_filter1d(y_hat_subset.mean(dim=[1]), sigma=10, axis=1))

                axs[k].plot(y_psth[:, neuron_plot_id], color='blue')
                [axs[k].plot(y_hat_psth[j, :, neuron_plot_id], color=colors(j), linewidth=0.5, alpha=0.5) for j in range(cfg.n_samples)]
                axs[k].axvline(130, linestyle='--', color='gray')

        fig.savefig(f'plots/dmfc_psth_predictions_acausal_id_{task_id}.pdf', bbox_inches='tight', transparent=True)
        plt.show()



if __name__ == '__main__':
    main()
