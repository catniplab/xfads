import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import pytorch_lightning as lightning

from hydra import compose, initialize
from xfads.plot_utils import plot_z_samples
from xfads.smoothers.lightning_trainers import LightningMonkeyReaching
from xfads.ssm_modules.prebuilt_models import create_xfads_poisson_log_link


def main():
    # at t=n_bins_bhv start forecast
    n_bins_bhv = 10

    torch.cuda.empty_cache()
    initialize(version_base=None, config_path="", job_name="monkey_reaching")
    cfg = compose(config_name="config")

    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    """data"""
    data_path = 'data/data_{split}_{bin_sz_ms}ms.pt'
    train_data = torch.load(data_path.format(split='train', bin_sz_ms=cfg.bin_sz_ms))
    val_data = torch.load(data_path.format(split='valid', bin_sz_ms=cfg.bin_sz_ms))
    test_data = torch.load(data_path.format(split='test', bin_sz_ms=cfg.bin_sz_ms))

    y_valid_obs = val_data['y_obs'].type(torch.float32).to(cfg.data_device)
    y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)
    y_test_obs = test_data['y_obs'].type(torch.float32).to(cfg.data_device)
    vel_valid = val_data['velocity'].type(torch.float32).to(cfg.data_device)
    vel_train = train_data['velocity'].type(torch.float32).to(cfg.data_device)
    vel_test = test_data['velocity'].type(torch.float32).to(cfg.data_device)
    n_trials, n_time_bins, n_neurons_obs = y_train_obs.shape
    n_time_bins_enc = train_data['n_time_bins_enc']

    y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, vel_train)
    y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, vel_valid)
    y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, vel_test)
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(y_test_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)

    """create ssm"""
    ssm = create_xfads_poisson_log_link(cfg, n_neurons_obs, train_dataloader, model_type='n')

    """lightning"""
    model_ckpt_path = 'results/noncausal_model.ckpt'
    seq_vae = LightningMonkeyReaching.load_from_checkpoint(model_ckpt_path, ssm=ssm, cfg=cfg,
                                                           n_time_bins_enc=n_time_bins_enc, n_time_bins_bhv=n_bins_bhv,
                                                           strict=False)
    """extract trained ssm from lightning module"""
    seq_vae.ssm = seq_vae.ssm.to(cfg.device)
    seq_vae.ssm.eval()

    z_s_train = []
    z_s_valid = []
    z_f_valid = []
    z_p_valid = []

    for batch in train_dataloader:
        loss, z, stats = seq_vae.ssm(batch[0], cfg.n_samples)
        z_s_train.append(z)

    for batch in valid_dataloader:
        z_f, stats = seq_vae.ssm.fast_filter_1_to_T(batch[0], cfg.n_samples)
        loss, z, stats = seq_vae.ssm(batch[0], cfg.n_samples)
        z_p = seq_vae.ssm.predict_forward(z_f[:, :, 10], cfg.n_samples)
        z_p = torch.cat([z_f[:, :, :10], z_p], dim=2)
        z_f_valid.append(z_f)
        z_p_valid.append(z_p)
        z_s_valid.append(z)

    U, S, V = torch.svd(seq_vae.ssm.likelihood_pdf.readout_fn[-1].weight.data)
    V = S.unsqueeze(-1) * V

    z_s_train = torch.cat(z_s_train, dim=1)
    z_s_valid = torch.cat(z_s_valid, dim=1)
    z_f_valid = torch.cat(z_f_valid, dim=1)
    z_p_valid = torch.cat(z_p_valid, dim=1)

    z_s_train = z_s_train[..., :cfg.n_latents_read] @ V
    z_s_test = z_s_valid[..., :cfg.n_latents_read] @ V
    z_f_test = z_f_valid[..., :cfg.n_latents_read] @ V
    z_p_test = z_p_valid[..., :cfg.n_latents_read] @ V

    """colors"""
    blues = cm.get_cmap("winter", z_s_test.shape[0])
    reds = cm.get_cmap("summer", z_s_test.shape[0])
    springs = cm.get_cmap("spring", z_s_test.shape[0])

    """plot sample regimes"""
    with torch.no_grad():
        trial_list = [28, 202, 8, 285]
        color_map_list = [blues, reds, springs]

        fig, axs = plt.subplots(len(trial_list), 1, figsize=(4, 4))
        plot_z_samples(fig, axs, z_s_test[:, trial_list, ..., :3], color_map_list)
        fig.savefig('plots/z_s_test.pdf', bbox_inches='tight', transparent=True)
        plt.show()

        fig, axs = plt.subplots(len(trial_list), 1, figsize=(4, 4))
        plot_z_samples(fig, axs, z_f_test[:, trial_list, ..., :3], color_map_list)
        fig.savefig('plots/z_f_test.pdf', bbox_inches='tight', transparent=True)
        plt.show()

        fig, axs = plt.subplots(len(trial_list), 1, figsize=(4, 4))
        [axs[i].axvline(10, linestyle='--', color='red') for i in range(len(trial_list))]
        plot_z_samples(fig, axs, z_p_test[:, trial_list, ..., :3], color_map_list)
        fig.savefig('plots/z_prd_test.pdf', bbox_inches='tight', transparent=True)
        plt.show()



if __name__ == '__main__':
    main()
