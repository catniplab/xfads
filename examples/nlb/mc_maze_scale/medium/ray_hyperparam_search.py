import os
import yaml
import torch
import tempfile
import dill as pickle
import torch.nn as nn
import dev.utils as utils
import torch.optim as optim
import ray.cloudpickle as pickle
import pytorch_lightning as lightning

from dev import prob_utils
from functools import partial
from omegaconf import OmegaConf
from hydra import compose, initialize
from dev.utils import evaluate_nlb_veloc_r2, get_updated_base_cfg

from dev.ssm_modules.likelihoods import PoissonLikelihood
from dev.ssm_modules.dynamics import DenseGaussianDynamics
from dev.ssm_modules.dynamics import DenseGaussianInitialCondition
from dev.smoothers.nonlinear_smoother import NonlinearFilter, LrSSMcoBPS
from dev.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn

from ray import tune
from ray import train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def load_data(cfg, data_path):
    """data"""
    train_data = torch.load(data_path.format(split='train', bin_sz_ms=cfg.bin_sz_ms))
    val_data = torch.load(data_path.format(split='valid', bin_sz_ms=cfg.bin_sz_ms))

    y_valid_obs = val_data['y_obs'].type(torch.float32).to(cfg.data_device)
    y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)
    vel_valid = val_data['velocity'].type(torch.float32).to(cfg.data_device)
    vel_train = train_data['velocity'].type(torch.float32).to(cfg.data_device)

    y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, vel_train)
    y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, vel_valid)
    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)

    data_metadata = {}
    data_metadata['n_neurons_obs'] = y_train_obs.shape[-1]
    data_metadata['n_neurons_enc'] = train_data['n_neurons_enc']
    data_metadata['n_time_bins_enc'] = train_data['n_time_bins_enc']

    return train_dataloader, valid_dataloader, data_metadata


def build_model(ray_cfg, n_neurons_obs, n_neurons_enc, n_time_bins_enc):
    cfg, _ = get_updated_base_cfg(ray_cfg)
    lightning.seed_everything(cfg.seed, workers=True)

    """likelihood pdf"""
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    C = utils.FanInLinear(cfg.n_latents_read, n_neurons_obs, device=cfg.device)
    readout_fn = torch.nn.Sequential(H, C)
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q = cfg.Q_init * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q, device=cfg.device)

    """initial condition"""
    Q0 = torch.ones(cfg.n_latents, device=cfg.device)
    m0 = torch.zeros(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m0, Q0, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents_read, cfg.n_hidden_backward, cfg.n_latents, cfg.rank_local,
                                            cfg.rank_backward, device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents_read, n_neurons_enc, cfg.n_hidden_local, cfg.n_latents,
                                      cfg.rank_local, device=cfg.device, dropout=cfg.p_local_dropout)
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

    """sequence vae"""
    ssm = LrSSMcoBPS(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder, local_encoder, nl_filter,
                     n_neurons_enc, n_time_bins_enc, device=cfg.device)

    return ssm


def train_ssm(ray_cfg, data_dir=None, print_frq=10):
    cfg, _ = get_updated_base_cfg(ray_cfg)
    lightning.seed_everything(cfg.seed, workers=True)

    """load data"""
    train_dataloader, valid_dataloader, data_metadata = load_data(cfg, data_dir)
    n_neurons_enc = data_metadata['n_neurons_enc']
    n_time_bins_enc = data_metadata['n_time_bins_enc']

    """build model, init bias"""
    ssm = build_model(ray_cfg, **data_metadata)
    ssm.likelihood_pdf.readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz)

    """set devices"""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            ssm = nn.DataParallel(ssm)

    ssm.to(device)
    optimizer = optim.Adam(ssm.parameters(), lr=1e-3)

    """load checkpoint if available"""
    if train.get_checkpoint():
        with train.get_checkpoint().as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                checkpoint_state = pickle.load(fp)

        start_epoch = checkpoint_state["epoch"]
        ssm.load_state_dict(checkpoint_state["ssm_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    """training loop"""
    for epoch in range(start_epoch, cfg.n_epochs):
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(train_dataloader, 0):
            ssm.train()
            y_obs, vel_obs = data
            y_obs, vel_obs = y_obs.to(device), vel_obs.to(device)

            optimizer.zero_grad()
            loss, z_s, stats = ssm(y_obs, cfg.n_samples, p_mask_a=cfg.p_mask_a, p_mask_apb=cfg.p_mask_apb,
                                   p_mask_y_in=cfg.p_mask_y_in, p_mask_b=cfg.p_mask_b, use_cd=cfg.use_cd)
            loss.backward()
            optimizer.step()

            # --- print statistics --- #
            running_loss += loss.item()
            epoch_steps += 1
            if i % print_frq == print_frq - 1:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        """validation loop"""
        val_loss = 0.
        val_bps_enc = 0.
        val_bps_hld = 0.
        val_steps = 0

        for i, data in enumerate(valid_dataloader, 0):
            ssm.eval()

            with torch.no_grad():
                ssm.eval()
                y_obs, vel_obs = data
                y_obs, vel_obs = y_obs.to(device), vel_obs.to(device)

                loss, z_s, stats = ssm(y_obs, cfg.n_samples)
                z_s_prd, stats_prd = ssm.predict(y_obs[..., :n_time_bins_enc, :n_neurons_enc], cfg.n_samples)

                val_loss += loss.cpu().numpy()
                val_bps_enc = prob_utils.bits_per_spike(stats_prd['log_rate'][..., :n_neurons_enc],
                                                    y_obs[..., :n_time_bins_enc, :n_neurons_enc].to(device)).cpu().numpy()
                val_bps_hld = prob_utils.bits_per_spike(stats_prd['log_rate'][..., n_neurons_enc:],
                                                    y_obs[..., :n_time_bins_enc, n_neurons_enc:].to(device)).cpu().numpy()

                val_steps += 1

        """get r2 scores"""
        r2_scores = evaluate_nlb_veloc_r2(cfg, ssm, train_dataloader, valid_dataloader, data_metadata, device)

        """log training/model state and save checkpoint"""
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:
                checkpoint_data = {
                    "epoch": epoch,
                    "ssm_state_dict": ssm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            metrics = {'val_loss': val_loss / val_steps,
                       'val_bps_enc': val_bps_enc / val_steps,
                       'val_bps_hld': val_bps_hld / val_steps,
                       'val_r2_veloc': r2_scores['r2_valid'],
                       }

            train.report(metrics, checkpoint=checkpoint)

    print("Finished Training")


def main():
    cwd = os.getcwd()
    data_path = cwd + '/data/data_{split}_{bin_sz_ms}ms.pt'
    initialize(version_base=None, config_path="", job_name="mc_maze_medium")
    cfg = compose(config_name="config")

    cpus_per_trial = 3
    gpus_per_trial = 0.1
    num_samples = cfg.n_ray_samples

    ray_cfg = {
        "base_cfg": cfg,
        "cwd": os.getcwd(),

        "p_mask_a": tune.uniform(0.0, 0.4),
        "p_mask_b": tune.uniform(0.0, 0.5),
        "p_mask_apb": tune.loguniform(0.0, 0.5),
        "p_mask_y_in": tune.loguniform(0.0, 0.5),
        "p_local_dropout": tune.loguniform(1e-3, 0.5),
        "rank_local": tune.randint(3, 30),
        "rank_backward": tune.randint(1, 20)
    }

    scheduler = ASHAScheduler(
        metric="val_bps_hld",
        mode="max",
        max_t=cfg.n_epochs,
        grace_period=cfg.n_epochs//2,
        reduction_factor=2,
    )

    result = tune.run(
        partial(train_ssm, data_dir=data_path),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=ray_cfg,
        num_samples=num_samples,
        scheduler=scheduler,
        keep_checkpoints_num=3)

    # --- save best configs --- #
    best_trial = result.get_best_trial("val_bps_hld", "max", "last")
    print(f"Best trial final validation heldout bps: {best_trial.last_result['val_bps_hld']}")

    metric_mode_map = {'val_loss': 'min',
                       'val_bps_hld': 'max',
                       'val_r2_veloc': 'max'}

    for metric, mode in metric_mode_map.items():
        with open(f"config_ray_{metric}.yaml", "w") as f:
            best_trial = result.get_best_trial(metric, mode, "last")
            base_cfg_best, _ = get_updated_base_cfg(best_trial.config)
            OmegaConf.save(base_cfg_best, f)

        best_trial = result.dataframe(metric, mode)
        best_trial.to_pickle(f'logs_ray/{metric}_df.pkl')


if __name__ == '__main__':
    main()
