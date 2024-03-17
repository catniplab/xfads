import math
import torch
import dev.utils as utils
import dev.prob_utils as prob_utils
import pytorch_lightning as lightning

from sklearn.linear_model import Ridge


class LightningNonlinearSSM(lightning.LightningModule):
    def __init__(self, ssm, cfg):
        super().__init__()

        self.ssm = ssm
        self.cfg = cfg

        self.n_samples = cfg.n_samples

        self.p_mask_a = cfg.p_mask_a
        self.p_mask_b = cfg.p_mask_b
        self.p_mask_apb = cfg.p_mask_apb
        self.p_mask_y_in = cfg.p_mask_y_in

        self.save_hyperparameters(ignore=['ssm', 'cfg'])

    def validation_step(self, batch, batch_idx):
        y = batch[0]

        loss, z_s, stats = self.ssm(y, self.n_samples)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch[0]

        p_mask_y_in_t = self.p_mask_y_in
        p_mask_b_t = self.p_mask_b * (1 + math.cos(2 * math.pi * self.current_epoch / 17.)) / 2.0
        p_mask_a_t = self.p_mask_a * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_apb_t = self.p_mask_apb * (1 + math.cos(2 * math.pi * self.current_epoch / 23.)) / 2.0

        loss, z_s, stats = self.ssm(y, self.n_samples, p_mask_a=p_mask_a_t, p_mask_apb=p_mask_apb_t,
                                    p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        log_Q_min = utils.softplus_inv(1e-3)
        log_Q_0_min = utils.softplus_inv(1e-1)

        self.ssm.dynamics_mod.log_Q.data = torch.clip(self.ssm.dynamics_mod.log_Q.data, min=log_Q_min)
        self.ssm.initial_c_pdf.log_Q_0.data = torch.clip(self.ssm.initial_c_pdf.log_Q_0.data, min=log_Q_0_min)


class LightningNlbNonlinearSSM(lightning.LightningModule):
    def __init__(self, ssm, cfg):
        super().__init__()

        self.ssm = ssm
        self.cfg = cfg

        self.l2_C = cfg.l2_C
        self.n_samples = cfg.n_samples

        self.use_cd = cfg.use_cd
        self.p_mask_a = cfg.p_mask_a
        self.p_mask_b = cfg.p_mask_b
        self.p_mask_apb = cfg.p_mask_apb
        self.p_mask_y_in = cfg.p_mask_y_in

        self.save_hyperparameters(ignore=['ssm', 'cfg'])

        self.valid_rates = []
        self.train_rates = []
        self.valid_veloc = []
        self.train_veloc = []

    def training_step(self, batch, batch_idx):
        y_obs = batch[0]
        n_neurons_enc = self.ssm.n_neurons_enc
        n_time_bins_enc = self.ssm.n_time_bins_enc

        l2_C = self.l2_C
        p_mask_y_in_t = self.p_mask_y_in

        p_mask_b_t = self.p_mask_b * (1 + math.cos(2 * math.pi * self.current_epoch / 17.)) / 2.0
        p_mask_a_t = self.p_mask_a * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_apb_t = self.p_mask_apb * (1 + math.cos(2 * math.pi * self.current_epoch / 23.)) / 2.0

        loss, z_s, stats = self.ssm(y_obs, self.n_samples, p_mask_a=p_mask_a_t, p_mask_apb=p_mask_apb_t,
                                    p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t, use_cd=self.use_cd, l2_C=l2_C)

        with torch.no_grad():
            self.ssm.eval()
            z_s_prd, stats_prd = self.ssm.predict(y_obs[..., :n_time_bins_enc, :n_neurons_enc],
                                                  self.n_samples, self.p_mask_y_in)
            self.ssm.train()

            bps_enc = prob_utils.bits_per_spike(stats_prd['log_rate'][..., :n_neurons_enc],
                                                y_obs[..., :n_time_bins_enc, :n_neurons_enc])
            bps_hld = prob_utils.bits_per_spike(stats_prd['log_rate'][..., n_neurons_enc:],
                                                y_obs[..., :n_time_bins_enc, n_neurons_enc:])
            self.train_rates.append(torch.exp(stats_prd['log_rate'][:, :n_time_bins_enc]))
            self.train_veloc.append(batch[1])

            self.log("train_bps_enc", bps_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_bps_hld", bps_hld, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_obs = batch[0]
        n_neurons_enc = self.ssm.n_neurons_enc
        n_time_bins_enc = self.ssm.n_time_bins_enc

        loss, z_s, stats = self.ssm(y_obs, self.n_samples)
        z_s_prd, stats_prd = self.ssm.predict(y_obs[..., :n_time_bins_enc, :n_neurons_enc],
                                              self.n_samples, self.p_mask_y_in)

        bps_enc = prob_utils.bits_per_spike(stats_prd['log_rate'][..., :n_neurons_enc],
                                            y_obs[..., :n_time_bins_enc, :n_neurons_enc])
        bps_hld = prob_utils.bits_per_spike(stats_prd['log_rate'][..., n_neurons_enc:],
                                            y_obs[..., :n_time_bins_enc, n_neurons_enc:])
        self.valid_rates.append(torch.exp(stats_prd['log_rate'][:, :n_time_bins_enc]))
        self.valid_veloc.append(batch[1])

        self.log("val_bps_hld", bps_hld, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_bps_enc", bps_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        log_Q_min = utils.softplus_inv(1e-3)
        log_Q_0_min = utils.softplus_inv(1e-1)

        self.ssm.dynamics_mod.log_Q.data = torch.clip(self.ssm.dynamics_mod.log_Q.data, min=log_Q_min)
        self.ssm.initial_c_pdf.log_Q_0.data = torch.clip(self.ssm.initial_c_pdf.log_Q_0.data, min=log_Q_0_min)

    def on_validation_epoch_end(self):
        if self.current_epoch < 1:
            self.valid_rates = []
            self.train_rates = []
            self.valid_veloc = []
            self.train_veloc = []
            return

        rates_train = torch.cat(self.train_rates, dim=0)
        rates_valid = torch.cat(self.valid_rates, dim=0)
        veloc_train = torch.cat(self.train_veloc, dim=0)
        veloc_valid = torch.cat(self.valid_veloc, dim=0)
        n_trials, n_time_bins, n_neurons = rates_train.shape

        clf = Ridge(alpha=0.01)
        # clf = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 0, 9)})

        clf.fit(rates_train.reshape(-1, n_neurons).cpu(), veloc_train.reshape(-1, 2).cpu())
        r2_train = clf.score(rates_train.reshape(-1, n_neurons).cpu(), veloc_train.reshape(-1, 2).cpu())
        r2_valid = clf.score(rates_valid.reshape(-1, n_neurons).cpu(), veloc_valid.reshape(-1, 2).cpu())

        self.valid_rates = []
        self.train_rates = []
        self.valid_veloc = []
        self.train_veloc = []

        self.log("train_veloc_r2", r2_train, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_veloc_r2", r2_valid, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

