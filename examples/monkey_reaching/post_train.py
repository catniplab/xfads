import torch
import pytorch_lightning as lightning

from hydra import compose, initialize
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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
    ssm = create_xfads_poisson_log_link(cfg, n_neurons_obs, train_dataloader)

    """lightning"""
    model_ckpt_path = 'results/noncausal_model.ckpt'
    seq_vae = LightningMonkeyReaching.load_from_checkpoint(model_ckpt_path, ssm=ssm, cfg=cfg,
                                                           n_time_bins_enc=n_time_bins_enc, n_time_bins_bhv=n_bins_bhv,
                                                           strict=False)
    """extract trained ssm from lightning module"""
    ssm = seq_vae.ssm


if __name__ == '__main__':
    main()
