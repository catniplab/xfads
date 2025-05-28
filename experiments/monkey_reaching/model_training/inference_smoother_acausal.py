import torch
import pytorch_lightning as lightning

from hydra import compose, initialize
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from xfads.smoothers.lightning_trainers import LightningMonkeyReaching
from xfads.ssm_modules.prebuilt_models import create_xfads_poisson_log_link_w_input


def main():
    # at t=n_bins_bhv start forecast
    n_bins_bhv = 50

    torch.cuda.empty_cache()
    initialize(version_base=None, config_path="", job_name="monkey_reaching")
    cfg = compose(config_name="config")

    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    """data"""
    data_path = "../data/data_{split}_{bin_sz_ms}ms_move_onset_time.pt"
    train_data = torch.load(data_path.format(split="train", bin_sz_ms=cfg.bin_sz_ms))
    val_data = torch.load(data_path.format(split="valid", bin_sz_ms=cfg.bin_sz_ms))
    test_data = torch.load(data_path.format(split="test", bin_sz_ms=cfg.bin_sz_ms))

    y_valid_obs = val_data["y_obs"].type(torch.float32).to(cfg.data_device)
    valid_input = val_data["input"].type(torch.float32).to(cfg.data_device)
    y_train_obs = train_data["y_obs"].type(torch.float32).to(cfg.data_device)
    train_input = train_data["input"].type(torch.float32).to(cfg.data_device)
    y_test_obs = test_data["y_obs"].type(torch.float32).to(cfg.data_device)
    test_input = test_data["input"].type(torch.float32).to(cfg.data_device)
    vel_valid = val_data["velocity"].type(torch.float32).to(cfg.data_device)
    vel_train = train_data["velocity"].type(torch.float32).to(cfg.data_device)
    vel_test = test_data["velocity"].type(torch.float32).to(cfg.data_device)
    n_trials, n_time_bins, n_neurons_obs = y_train_obs.shape
    n_time_bins_enc = train_data["n_time_bins_enc"]
    n_inputs = valid_input.shape[-1]

    y_train_dataset = torch.utils.data.TensorDataset(
        y_train_obs, vel_train, train_input
    )
    y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, vel_valid, valid_input)
    y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, vel_test, test_input)
    train_dataloader = torch.utils.data.DataLoader(
        y_train_dataset, batch_size=cfg.batch_sz, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        y_test_dataset, batch_size=y_valid_obs.shape[0], shuffle=False
    )

    """create ssm"""
    ssm = create_xfads_poisson_log_link_w_input(
        cfg, n_neurons_obs, n_inputs, train_dataloader, model_type="c"
    )

    # for batch in train_dataloader:
    #     print('1')
    #     y_obs = batch[0]
    #     x_obs = batch[1]
    #     u_obs = batch[-1]
    #     n_time_bins_enc = n_time_bins_enc
    #     loss, z_s, stats = ssm(y_obs[:, :n_time_bins_enc], u_obs[:, :n_time_bins_enc], cfg.n_samples)

    """lightning"""
    model_ckpt_path = "ckpts/noncausal_model.ckpt"
    # seq_vae = LightningMonkeyReaching(ssm, cfg, n_time_bins_enc, n_bins_bhv)
    seq_vae = LightningMonkeyReaching.load_from_checkpoint(
        model_ckpt_path,
        ssm=ssm,
        cfg=cfg,
        n_time_bins_enc=n_time_bins_enc,
        n_time_bins_bhv=n_bins_bhv,
        use_input=True,
        strict=False,
    )

    csv_logger = CSVLogger(
        "logs/smoother/acausal/",
        name=f"sd_{cfg.seed}_r_y_{cfg.rank_local}_r_b_{cfg.rank_backward}",
        version="smoother_acausal",
    )
    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="r2_valid_enc",
        mode="max",
        dirpath="ckpts/smoother/acausal/",
        save_last=True,
        filename="{epoch:0}_{valid_loss:0.2f}_{r2_valid_enc:0.2f}_{r2_valid_bhv:0.2f}_{valid_bps_enc:0.2f}",
    )

    trainer = lightning.Trainer(
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.0,
        default_root_dir="lightning/",
        callbacks=[ckpt_callback],
        logger=csv_logger,
    )

    trainer.fit(
        model=seq_vae,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    torch.save(
        ckpt_callback.best_model_path, "ckpts/smoother/acausal/best_model_path.pt"
    )
    trainer.test(dataloaders=test_dataloader, ckpt_path="last")


if __name__ == "__main__":
    main()
