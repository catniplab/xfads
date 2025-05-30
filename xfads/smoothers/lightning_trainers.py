import math
import time
import copy
import torch
import skorch
import xfads.utils as utils
import xfads.prob_utils as prob_utils
import pytorch_lightning as lightning

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.lr_gamma_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    def training_step(self, batch, batch_idx):
        y = batch[0]

        p_mask_y_in_t = self.p_mask_y_in
        p_mask_b_t = self.p_mask_b * (1 + math.cos(2 * math.pi * self.current_epoch / 17.)) / 2.0
        p_mask_a_t = self.p_mask_a * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_apb_t = self.p_mask_apb * (1 + math.cos(2 * math.pi * self.current_epoch / 23.)) / 2.0

        t_start = time.time()
        loss, z_s, stats = self.ssm(y, self.n_samples, p_mask_a=p_mask_a_t, p_mask_apb=p_mask_apb_t,
                                    p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t)
        t_forward = time.time() - t_start

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[0]
        
        with torch.no_grad():
            loss, z_s, stats = self.ssm(y, self.n_samples)
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        return loss

    def test_step(self, batch, batch_idx):
        y = batch[0]

        with torch.no_grad():
            loss, z_s, stats = self.ssm(y, self.n_samples)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        with torch.no_grad():
            log_Q_min = utils.softplus_inv(1e-3)
            log_Q_0_min = utils.softplus_inv(1e-2)

            self.ssm.dynamics_mod.log_Q.data = torch.clip(self.ssm.dynamics_mod.log_Q.data, min=log_Q_min)
            self.ssm.initial_c_pdf.log_Q_0.data = torch.clip(self.ssm.initial_c_pdf.log_Q_0.data, min=log_Q_0_min)


class LightningNonlinearSSMwithInput(LightningNonlinearSSM):
    def __init__(self, ssm, cfg):
        super(LightningNonlinearSSMwithInput).__init__(ssm, cfg)

    def training_step(self, batch, batch_idx):
        y = batch[0]
        u = batch[1]

        p_mask_y_in_t = self.p_mask_y_in
        p_mask_b_t = self.p_mask_b * (1 + math.cos(2 * math.pi * self.current_epoch / 17.)) / 2.0
        p_mask_a_t = self.p_mask_a * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_apb_t = self.p_mask_apb * (1 + math.cos(2 * math.pi * self.current_epoch / 23.)) / 2.0

        t_start = time.time()
        loss, z_s, stats = self.ssm(y, u, self.n_samples, p_mask_a=p_mask_a_t, p_mask_apb=p_mask_apb_t,
                                    p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t)
        t_forward = time.time() - t_start

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[0]
        u = batch[1]
        
        with torch.no_grad():
            loss, z_s, stats = self.ssm(y, u, self.n_samples)
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        y = batch[0]
        u = batch[1]

        with torch.no_grad():
            loss, z_s, stats = self.ssm(y, u, self.n_samples)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss


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
        
        with torch.no_grad():
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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.lr_gamma_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        with torch.no_grad():
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
        
        with torch.no_grad():
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


class LightningMonkeyReaching(lightning.LightningModule):
    def __init__(self, ssm, cfg, n_time_bins_enc, bin_prd_start, is_svae=False, use_input=False):
        super().__init__()

        self.ssm = ssm
        self.cfg = cfg
        self.is_svae = is_svae
        self.use_input = use_input
        self.n_samples = cfg.n_samples

        self.use_cd = cfg.use_cd
        self.p_mask_a = cfg.p_mask_a
        self.p_mask_b = cfg.p_mask_b
        self.p_mask_apb = cfg.p_mask_apb
        self.p_mask_y_in = cfg.p_mask_y_in

        self.n_time_bins_enc = n_time_bins_enc
        self.bin_prd_start = bin_prd_start

        self.save_hyperparameters(ignore=['ssm', 'cfg'])

        self.train_rates_enc = []
        self.valid_rates_enc = []
        self.valid_rates_prd = []
        self.train_veloc_enc = []
        self.valid_veloc_enc = []
        self.valid_veloc_prd = []

        self.best_r2_enc = -1.0
        self.best_r2_prd = -1.0
        self.best_clf_enc = None
        self.best_clf_prd = None
        self.best_ssm_enc = None
        self.best_ssm_prd = None

    def training_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc

        p_mask_y_in_t = self.p_mask_y_in
        p_mask_a_t = min(self.p_mask_a, self.current_epoch * self.p_mask_a / 300)
        p_mask_a_t = p_mask_a_t * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_b_t = self.p_mask_b * (1 + math.cos(2 * math.pi * self.current_epoch / 17.)) / 2.0
        p_mask_apb_t = self.p_mask_apb * (1 + math.cos(2 * math.pi * self.current_epoch / 23.)) / 2.0

        t_start = time.time()

        if self.use_input:
            u_obs = batch[-1]
            loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], u_obs[:, :n_time_bins_enc], self.n_samples,
                                        p_mask_a=p_mask_a_t, p_mask_apb=p_mask_apb_t, p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t)
        else:
            loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples,
                                        p_mask_a=p_mask_a_t, p_mask_apb=p_mask_apb_t, p_mask_y_in=p_mask_y_in_t,
                                        p_mask_b=p_mask_b_t)

        t_forward = time.time() - t_start

        with torch.no_grad():
            rate_enc_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_s)).mean(dim=0)
            bps_enc = prob_utils.bits_per_spike(torch.log(rate_enc_hat), y_obs[:, :n_time_bins_enc])

            self.train_rates_enc.append(rate_enc_hat)
            self.train_veloc_enc.append(x_obs[:, :n_time_bins_enc])

            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train_bps_enc", bps_enc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc
        
        with torch.no_grad():
            if self.use_input:
                u_obs = batch[-1]
                loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], u_obs[:, :n_time_bins_enc], self.n_samples)
                _, z_s_prd_init, stats_enc = self.ssm(y_obs[:, :self.bin_prd_start], u_obs[:, :n_time_bins_enc], self.n_samples)
                z_p_prd = self.ssm.predict_forward(z_s_prd_init[:, :, -1], u_obs[:, self.bin_prd_start:], n_time_bins_enc - self.bin_prd_start)
            else:
                loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples)
                _, z_s_prd_init, stats_enc = self.ssm(y_obs[:, :self.bin_prd_start], self.n_samples)
                z_p_prd = self.ssm.predict_forward(z_s_prd_init[:, :, -1], n_time_bins_enc - self.bin_prd_start)

            rate_enc_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_s)).mean(
                dim=0)
            rate_prd_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_p_prd)).mean(
                dim=0)

            bps_enc = prob_utils.bits_per_spike(torch.log(rate_enc_hat), y_obs[:, :n_time_bins_enc])
            bps_prd = prob_utils.bits_per_spike(torch.log(rate_prd_hat), y_obs[:, self.bin_prd_start:n_time_bins_enc])

            self.valid_rates_enc.append(rate_enc_hat)
            self.valid_rates_prd.append(rate_prd_hat)
            self.valid_veloc_enc.append(x_obs[:, :n_time_bins_enc])

            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("valid_bps_prd", bps_prd, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("valid_bps_enc", bps_enc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            
        return loss

    def test_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc
        
        with torch.no_grad():
            if self.use_input:
                u_obs = batch[-1]
                loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], u_obs[:, :n_time_bins_enc], self.n_samples)
                _, z_prd_init, _ = self.best_ssm_prd(y_obs[:, :self.bin_prd_start], u_obs[:, :self.bin_prd_start], self.n_samples)
                z_p_prd = self.best_ssm_prd.predict_forward(z_prd_init[:, :, -1], u_obs[:, self.bin_prd_start:], n_time_bins_enc - self.bin_prd_start)
                _, z_enc, _ = self.best_ssm_enc(y_obs[:, :n_time_bins_enc], u_obs[:, :n_time_bins_enc], self.n_samples)
            else:
                loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples)
                _, z_prd_init, _ = self.best_ssm_prd(y_obs[:, :self.bin_prd_start], self.n_samples)
                z_p_prd = self.best_ssm_prd.predict_forward(z_prd_init[:, :, -1], n_time_bins_enc - self.bin_prd_start)
                _, z_enc, _ = self.best_ssm_enc(y_obs[:, :n_time_bins_enc], self.n_samples)

            rate_enc_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_s)).mean(
                dim=0)

            r2_test_enc = self.best_clf_enc.score(rate_enc_hat.reshape(-1, rate_enc_hat.shape[-1]).cpu(),
                                                  x_obs[:, :n_time_bins_enc].reshape(-1, 2).cpu())

            rate_prd_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_p_prd)).mean(
                dim=0)

            r2_test_prd = self.best_clf_prd.score(rate_prd_hat.reshape(-1, rate_prd_hat.shape[-1]).cpu(),
                                                  x_obs[:, self.bin_prd_start: n_time_bins_enc].reshape(-1, 2).cpu())

            bps_enc = prob_utils.bits_per_spike(torch.log(rate_enc_hat), y_obs[:, :n_time_bins_enc])
            bps_prd = prob_utils.bits_per_spike(torch.log(rate_prd_hat), y_obs[:, self.bin_prd_start:n_time_bins_enc])

            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_bps_prd", bps_prd, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("test_bps_enc", bps_enc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("r2_test_enc", r2_test_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("r2_test_prd", r2_test_prd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.lr_gamma_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        with torch.no_grad():
            log_Q_min = utils.softplus_inv(1e-3)
            log_Q_0_min = utils.softplus_inv(1e-1)
        
            self.ssm.dynamics_mod.log_Q.data = torch.clip(self.ssm.dynamics_mod.log_Q.data, min=log_Q_min)
            self.ssm.initial_c_pdf.log_Q_0.data = torch.clip(self.ssm.initial_c_pdf.log_Q_0.data, min=log_Q_0_min)

            if self.is_svae:
                F = self.ssm.dynamics_mod.mean_fn.weight.data
                U, S, VmT = torch.linalg.svd(F, full_matrices=True)
                self.ssm.dynamics_mod.mean_fn.weight.data = (U * S.clip(max=0.98)) @ VmT

    def on_validation_epoch_end(self):
        if self.current_epoch == 0:
            self.log("r2_valid_enc", -1., on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("r2_valid_prd", -1., on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("r2_train_enc", -1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        elif self.current_epoch > 1 and self.current_epoch % 3 == 0:
            with torch.no_grad():
                n_time_bins_enc = self.n_time_bins_enc
                train_rates_enc = torch.cat(self.train_rates_enc, dim=0)
                valid_rates_enc = torch.cat(self.valid_rates_enc, dim=0)
                valid_rates_prd = torch.cat(self.valid_rates_prd, dim=0)

                train_veloc_enc = torch.cat(self.train_veloc_enc, dim=0)
                valid_veloc_enc = torch.cat(self.valid_veloc_enc, dim=0)
                valid_veloc_prd = valid_veloc_enc[:, self.bin_prd_start:self.n_time_bins_enc]
                _, n_time_bins, n_neurons = train_rates_enc.shape

                clf = Ridge(alpha=0.01)
                clf.fit(train_rates_enc.reshape(-1, n_neurons).cpu(), train_veloc_enc.reshape(-1, 2).cpu())

                r2_train_enc = clf.score(train_rates_enc.reshape(-1, n_neurons).cpu(), train_veloc_enc.reshape(-1, 2).cpu())
                r2_valid_enc = clf.score(valid_rates_enc.reshape(-1, n_neurons).cpu(), valid_veloc_enc.reshape(-1, 2).cpu())
                r2_valid_prd = clf.score(valid_rates_prd.reshape(-1, n_neurons).cpu(), valid_veloc_prd.reshape(-1, 2).cpu())

                if r2_valid_enc >= self.best_r2_enc:
                    print(f'current_epoch: {self.current_epoch}')
                    self.best_clf_enc = copy.deepcopy(clf)
                    self.best_ssm_enc = copy.deepcopy(self.ssm)
                if r2_valid_prd >= self.best_r2_prd:
                    self.best_clf_prd = copy.deepcopy(clf)
                    self.best_ssm_prd = copy.deepcopy(self.ssm)

                self.log("r2_train_enc", r2_train_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log("r2_valid_enc", r2_valid_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log("r2_valid_prd", r2_valid_prd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_rates_enc = []
        self.valid_rates_enc = []
        self.valid_rates_prd = []

        self.train_veloc_enc = []
        self.valid_veloc_enc = []


class LightningDMFCRSG(lightning.LightningModule):
    def __init__(self, ssm, cfg, n_time_bins_enc, n_time_bins_prd, is_svae=False):
        super().__init__()

        self.ssm = ssm
        self.cfg = cfg
        self.is_svae = is_svae
        self.n_samples = cfg.n_samples

        self.use_cd = cfg.use_cd
        self.p_mask_a = cfg.p_mask_a
        self.p_mask_b = cfg.p_mask_b
        self.p_mask_apb = cfg.p_mask_apb
        self.p_mask_y_in = cfg.p_mask_y_in

        self.n_time_bins_enc = n_time_bins_enc
        self.n_time_bins_prd = n_time_bins_prd

        self.save_hyperparameters(ignore=['ssm', 'cfg'])

        self.best_bps_enc = -1.0
        self.best_bps_prd = -1.0
        self.best_ssm_enc = None
        self.best_ssm_prd = None

    def training_step(self, batch, batch_idx):
        y_obs = batch[0]
        n_time_bins_enc = self.n_time_bins_enc

        p_mask_y_in_t = self.p_mask_y_in
        p_mask_a_t = min(self.p_mask_a, self.current_epoch * self.p_mask_a / 100)
        p_mask_a_t = p_mask_a_t * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_b_t = self.p_mask_b * (1 + math.cos(2 * math.pi * self.current_epoch / 17.)) / 2.0
        p_mask_apb_t = self.p_mask_apb * (1 + math.cos(2 * math.pi * self.current_epoch / 23.)) / 2.0

        t_start = time.time()
        loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples, p_mask_a=p_mask_a_t,
                                    p_mask_apb=p_mask_apb_t, p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t)

        t_forward = time.time() - t_start

        with torch.no_grad():
            rate_enc_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_s)).mean(dim=0)
            bps_enc = prob_utils.bits_per_spike(torch.log(rate_enc_hat), y_obs[:, :n_time_bins_enc])

            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train_bps_enc", bps_enc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_obs = batch[0]
        n_time_bins_enc = self.n_time_bins_enc
        
        with torch.no_grad():
            loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples)
            _, z_s_prd_init, stats_enc = self.ssm(y_obs[:, :self.n_time_bins_prd], self.n_samples)
            z_p_prd = self.ssm.predict_forward(z_s_prd_init[:, :, -1], n_time_bins_enc - self.n_time_bins_prd)

            rate_enc_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_s)).mean(
                dim=0)
            rate_prd_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_p_prd)).mean(
                dim=0)

            bps_enc = prob_utils.bits_per_spike(torch.log(rate_enc_hat), y_obs[:, :n_time_bins_enc])
            bps_prd = prob_utils.bits_per_spike(torch.log(rate_prd_hat), y_obs[:, self.n_time_bins_prd:n_time_bins_enc])

            if bps_enc >= self.best_bps_enc:
                print(f'current_epoch: {self.current_epoch}')
                self.best_ssm_enc = copy.deepcopy(self.ssm)
            if bps_prd >= self.best_bps_prd:
                self.best_ssm_prd = copy.deepcopy(self.ssm)

            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("valid_bps_enc", bps_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("valid_bps_prd", bps_prd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        return loss

    def test_step(self, batch, batch_idx):
        y_obs = batch[0]
        n_time_bins_enc = self.n_time_bins_enc
        
        with torch.no_grad():
            loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples)
            _, z_enc, _ = self.best_ssm_enc(y_obs[:, :n_time_bins_enc], self.n_samples)
            rate_enc_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_s)).mean(
                dim=0)

            _, z_prd_init, _ = self.best_ssm_prd(y_obs[:, :self.n_time_bins_prd], self.n_samples)
            z_p_prd = self.best_ssm_prd.predict_forward(z_prd_init[:, :, -1], n_time_bins_enc - self.n_time_bins_prd)
            rate_prd_hat = self.ssm.likelihood_pdf.delta * torch.exp(self.ssm.likelihood_pdf.readout_fn(z_p_prd)).mean(
                dim=0)

            bps_enc = prob_utils.bits_per_spike(torch.log(rate_enc_hat), y_obs[:, :n_time_bins_enc])
            bps_prd = prob_utils.bits_per_spike(torch.log(rate_prd_hat), y_obs[:, self.n_time_bins_prd:n_time_bins_enc])

            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_bps_prd", bps_prd, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("test_bps_enc", bps_enc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'valid_bps_enc', 'frequency': 5}}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        with torch.no_grad():
            log_Q_min = utils.softplus_inv(1e-3)
            log_Q_0_min = utils.softplus_inv(1e-1)
        
            self.ssm.dynamics_mod.log_Q.data = torch.clip(self.ssm.dynamics_mod.log_Q.data, min=log_Q_min)
            self.ssm.initial_c_pdf.log_Q_0.data = torch.clip(self.ssm.initial_c_pdf.log_Q_0.data, min=log_Q_0_min)

            if self.is_svae:
                F = self.ssm.dynamics_mod.mean_fn.weight.data
                U, S, VmT = torch.linalg.svd(F, full_matrices=True)
                self.ssm.dynamics_mod.mean_fn.weight.data = (U * S.clip(max=0.99)) @ VmT


class LightningPendulum(lightning.LightningModule):
    def __init__(self, ssm, cfg, n_time_bins_enc):
        # z_true[:, :, 0] = cosPhi
        # z_true[:, :, 1] = sinPhi
        # z_true[:, :, 2] = phiDot
        # z_true[:, :, 3] = -phiDot * np.sin(phi)
        super().__init__()

        self.ssm = ssm
        self.cfg = cfg
        self.n_samples = cfg.n_samples

        self.use_cd = cfg.use_cd
        self.p_mask_a = cfg.p_mask_a
        self.p_mask_b = cfg.p_mask_b
        self.p_mask_apb = cfg.p_mask_apb
        self.p_mask_y_in = cfg.p_mask_y_in
        self.n_time_bins_enc = n_time_bins_enc

        self.save_hyperparameters(ignore=['ssm', 'cfg'])

        self.valid_z = []
        self.train_z = []
        self.valid_veloc = []
        self.train_veloc = []

        self.best_r2_enc = -1.0
        self.best_r2_prd = -1.0
        self.best_clf_enc = None
        self.best_clf_prd = None
        self.best_ssm_enc = None
        self.best_ssm_prd = None

    def training_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc

        p_mask_y_in_t = self.p_mask_y_in
        p_mask_a_t = min(self.p_mask_a, self.current_epoch * self.p_mask_a / 1000)
        p_mask_a_t = p_mask_a_t * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_b_t = self.p_mask_b * (1 + math.cos(2 * math.pi * self.current_epoch / 17.)) / 2.0
        p_mask_apb_t = self.p_mask_apb * (1 + math.cos(2 * math.pi * self.current_epoch / 23.)) / 2.0

        t_start = time.time()
        loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples, p_mask_a=p_mask_a_t,
                                    p_mask_apb=p_mask_apb_t, p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t)

        t_forward = time.time() - t_start
        self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            self.train_z.append(z_s)
            theta_dot = x_obs[..., 2] / x_obs[..., 0]
            self.train_veloc.append(theta_dot[..., None])

            z_hat = self.ssm.predict_forward(z_s[:, :, -1], y_obs.shape[1] - n_time_bins_enc)
            y_hat = self.ssm.likelihood_pdf.readout_fn(z_hat)
            mse = (y_hat - y_obs[:, n_time_bins_enc:]).pow(2).mean()

            self.log('train_mse', mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc
        n_time_bins_prd = y_obs.shape[1] - n_time_bins_enc
        
        with torch.no_grad():
            loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples)
            z_prd = self.ssm.predict_forward(z_s[:, :, -1], n_time_bins_prd)

            theta_dot = x_obs[..., 2] / x_obs[..., 0]
            self.valid_veloc.append(theta_dot[..., None])

            z_hat = self.ssm.predict_forward(z_s[:, :, -1], y_obs.shape[1] - n_time_bins_enc)
            y_hat = self.ssm.likelihood_pdf.readout_fn(z_hat)
            mse = (y_hat - y_obs[:, n_time_bins_enc:]).pow(2).mean()

            self.valid_z.append(torch.cat([z_s, z_prd], dim=2))
            self.log("valid_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        return loss

    def test_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc
        n_time_bins_prd = y_obs.shape[1] - n_time_bins_enc
        theta_dot = x_obs[..., 2] / x_obs[..., 0]
        
        with torch.no_grad():
            loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples)
            n_latents = z_s.shape[-1]

            z_hat = self.ssm.predict_forward(z_s[:, :, -1], y_obs.shape[1] - n_time_bins_enc)
            y_hat = self.ssm.likelihood_pdf.readout_fn(z_hat)
            mse = (y_hat - y_obs[:, n_time_bins_enc:]).pow(2).mean()

            _, z_s, _ = self.best_ssm_enc(y_obs[:, :n_time_bins_enc], self.n_samples)
            r2_test_enc = self.best_clf_enc.score(z_s.mean(dim=0).reshape(-1, n_latents).cpu(),
                                                   theta_dot[:, :n_time_bins_enc].reshape(-1, 1).cpu())

            _, z_prd_init, _ = self.best_ssm_prd(y_obs[:, :n_time_bins_enc], self.n_samples)
            z_prd = self.best_ssm_prd.predict_forward(z_prd_init[:, :, -1], n_time_bins_prd)
            r2_test_prd = self.best_clf_prd.score(z_prd.mean(dim=0).reshape(-1, n_latents).cpu(),
                                                   theta_dot[:, n_time_bins_enc:].reshape(-1, 1).cpu())

            self.valid_z = []
            self.train_z = []
            self.valid_veloc = []
            self.train_veloc = []

            self.log("test_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("r2_test_enc", r2_test_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("r2_test_prd", r2_test_prd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.lr_gamma_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        with torch.no_grad():
            log_Q_min = utils.softplus_inv(1e-3)
            log_Q_0_min = utils.softplus_inv(1e-1)

            self.ssm.dynamics_mod.log_Q.data = torch.clip(self.ssm.dynamics_mod.log_Q.data, min=log_Q_min)
            self.ssm.initial_c_pdf.log_Q_0.data = torch.clip(self.ssm.initial_c_pdf.log_Q_0.data, min=log_Q_0_min)

    def on_validation_epoch_end(self):
        if self.current_epoch > 0:
            n_time_bins_enc = self.n_time_bins_enc
            m_train = torch.cat(self.train_z, dim=1).mean(dim=0)
            n_train_trials = m_train.shape[0]

            # reduce number of regressor trials for training speed
            train_trial_dx_regress = torch.randperm(n_train_trials)[:n_train_trials//2]

            m_valid = torch.cat(self.valid_z, dim=1).mean(dim=0)
            veloc_train = torch.cat(self.train_veloc, dim=0)
            veloc_valid = torch.cat(self.valid_veloc, dim=0)
            n_trials, n_time_bins, n_latents = m_train.shape
            n_veloc = veloc_train.shape[-1]

            # clf = MLPRegressor(hidden_layer_sizes=(100,), alpha=1e-1)
            # clf.fit(m_train[train_trial_dx_regress, :n_time_bins_enc].reshape(-1, n_latents).cpu(),
            #         veloc_train[train_trial_dx_regress, :n_time_bins_enc].reshape(-1, n_veloc).cpu())
            # r2_valid_enc = clf.score(m_valid[:, :n_time_bins_enc].reshape(-1, n_latents).cpu(),
            #                          veloc_valid[:, :n_time_bins_enc].reshape(-1, n_veloc).cpu())
            # r2_valid_prd = clf.score(m_valid[:, n_time_bins_enc:].reshape(-1, n_latents).cpu(),
            #                          veloc_valid[:, n_time_bins_enc:].reshape(-1, n_veloc).cpu())

            clf = skorch.regressor.NeuralNetRegressor(torch.nn.Sequential(torch.nn.Linear(n_latents, 100,
                                                                                          device=self.cfg.device),
                                                                          torch.nn.SiLU(),
                                                                          torch.nn.Linear(100, n_veloc,
                                                                                          device=self.cfg.device)),
                                                      max_epochs=100, device=self.cfg.device,
                                                      optimizer=torch.optim.Adam,
                                                      optimizer__weight_decay=1e-3, optimizer__lr=1e-3, verbose=0)

            # clf.fit(m_train[train_trial_dx_regress, :n_time_bins_enc].reshape(-1, n_latents),
            #         veloc_train[train_trial_dx_regress, :n_time_bins_enc].reshape(-1, n_veloc))
            clf.fit(m_train[train_trial_dx_regress, :n_time_bins_enc],
                    veloc_train[train_trial_dx_regress, :n_time_bins_enc])

            # try:
            r2_valid_enc = clf.score(m_valid[:, :n_time_bins_enc].reshape(-1, n_latents).detach().cpu(),
                                     veloc_valid[:, :n_time_bins_enc].reshape(-1, n_veloc).detach().cpu())
            r2_valid_prd = clf.score(m_valid[:, n_time_bins_enc:].reshape(-1, n_latents).detach().cpu(),
                                     veloc_valid[:, n_time_bins_enc:].reshape(-1, n_veloc).detach().cpu())
            # except:
            #     r2_valid_enc = -1.
            #     r2_valid_prd = -1.

            if r2_valid_enc >= self.best_r2_enc:
                self.best_clf_enc = copy.deepcopy(clf)
                self.best_ssm_enc = copy.deepcopy(self.ssm)
                self.best_r2_enc = r2_valid_enc
            if r2_valid_prd >= self.best_r2_prd:
                self.best_clf_prd = copy.deepcopy(clf)
                self.best_ssm_prd = copy.deepcopy(self.ssm)
                self.best_r2_prd = r2_valid_prd

        # except:
        else:
            r2_valid_enc = -1.
            r2_valid_prd = -1.

        self.log("r2_valid_enc", r2_valid_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("r2_valid_prd", r2_valid_prd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.valid_z = []
        self.train_z = []
        self.valid_veloc = []
        self.train_veloc = []


class LightningBouncingBall(lightning.LightningModule):
    def __init__(self, ssm, cfg, n_time_bins_enc, is_svae=False):
        # x[n, :, 0] = x_loc
        # x[n, :, 1] = y_loc
        super().__init__()

        self.ssm = ssm
        self.cfg = cfg
        self.is_svae = is_svae
        self.n_samples = cfg.n_samples

        self.use_cd = cfg.use_cd
        self.p_mask_a = cfg.p_mask_a
        self.p_mask_b = cfg.p_mask_b
        self.p_mask_apb = cfg.p_mask_apb
        self.p_mask_y_in = cfg.p_mask_y_in
        self.n_time_bins_enc = n_time_bins_enc

        self.save_hyperparameters(ignore=['ssm', 'cfg'])

        self.valid_z = []
        self.train_z = []
        self.valid_veloc = []
        self.train_veloc = []

        self.best_r2_enc = -1.0
        self.best_r2_prd = -1.0
        self.best_clf_enc = None
        self.best_clf_prd = None
        self.best_ssm_enc = None
        self.best_ssm_prd = None

    def training_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc

        p_mask_y_in_t = self.p_mask_y_in
        p_mask_a_t = min(self.p_mask_a, self.current_epoch * self.p_mask_a / 1000)
        p_mask_a_t = p_mask_a_t * (1 + math.cos(2 * math.pi * self.current_epoch / 20.)) / 2.0
        p_mask_b_t = self.p_mask_b * (1 + math.cos(2 * math.pi * self.current_epoch / 17.)) / 2.0
        p_mask_apb_t = self.p_mask_apb * (1 + math.cos(2 * math.pi * self.current_epoch / 23.)) / 2.0

        t_start = time.time()
        loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples, p_mask_a=p_mask_a_t,
                                    p_mask_apb=p_mask_apb_t, p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t)

        t_forward = time.time() - t_start
        self.log("time_forward", t_forward, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            self.train_z.append(z_s)
            self.train_veloc.append(x_obs)

            z_hat = self.ssm.predict_forward(z_s[:, :, -1], y_obs.shape[1] - n_time_bins_enc)
            y_hat = self.ssm.likelihood_pdf.readout_fn(z_hat)
            mse = (y_hat - y_obs[:, n_time_bins_enc:]).pow(2).mean()

            self.log('train_mse', mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc
        n_time_bins_prd = y_obs.shape[1] - n_time_bins_enc

        with torch.no_grad():
            loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples)
            z_prd = self.ssm.predict_forward(z_s[:, :, -1], n_time_bins_prd)

            # theta_dot = x_obs[..., 2] / x_obs[..., 0]
            self.valid_veloc.append(x_obs)

            z_hat = self.ssm.predict_forward(z_s[:, :, -1], y_obs.shape[1] - n_time_bins_enc)
            y_hat = self.ssm.likelihood_pdf.readout_fn(z_hat)
            mse = (y_hat - y_obs[:, n_time_bins_enc:]).pow(2).mean()

            self.valid_z.append(torch.cat([z_s, z_prd], dim=2))
            self.log("valid_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        y_obs = batch[0]
        x_obs = batch[1]
        n_time_bins_enc = self.n_time_bins_enc
        n_time_bins_prd = y_obs.shape[1] - n_time_bins_enc
        # theta_dot = x_obs[..., 2] / x_obs[..., 0]
        
        with torch.no_grad():
            loss, z_s, stats = self.ssm(y_obs[:, :n_time_bins_enc], self.n_samples)
            n_latents = z_s.shape[-1]

            z_hat = self.ssm.predict_forward(z_s[:, :, -1], y_obs.shape[1] - n_time_bins_enc)
            y_hat = self.ssm.likelihood_pdf.readout_fn(z_hat)
            mse = (y_hat - y_obs[:, n_time_bins_enc:]).pow(2).mean()

            _, z_s, _ = self.best_ssm_enc(y_obs[:, :n_time_bins_enc], self.n_samples)
            r2_test_enc = self.best_clf_enc.score(z_s.mean(dim=0).reshape(-1, n_latents).cpu(),
                                                  x_obs[:, :n_time_bins_enc].reshape(-1, 2).cpu())

            _, z_prd_init, _ = self.best_ssm_prd(y_obs[:, :n_time_bins_enc], self.n_samples)
            z_prd = self.best_ssm_prd.predict_forward(z_prd_init[:, :, -1], n_time_bins_prd)
            r2_test_prd = self.best_clf_prd.score(z_prd.mean(dim=0).reshape(-1, n_latents).cpu(),
                                                  x_obs[:, n_time_bins_enc:].reshape(-1, 2).cpu())

            self.valid_z = []
            self.train_z = []
            self.valid_veloc = []
            self.train_veloc = []

            self.log("test_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("r2_test_enc", r2_test_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("r2_test_prd", r2_test_prd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.lr_gamma_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        with torch.no_grad():
            log_Q_min = utils.softplus_inv(1e-3)
            log_Q_0_min = utils.softplus_inv(1e-1)
        
            self.ssm.dynamics_mod.log_Q.data = torch.clip(self.ssm.dynamics_mod.log_Q.data, min=log_Q_min)
            self.ssm.initial_c_pdf.log_Q_0.data = torch.clip(self.ssm.initial_c_pdf.log_Q_0.data, min=log_Q_0_min)

            if self.is_svae:
                F = self.ssm.dynamics_mod.mean_fn.weight.data
                U, S, VmT = torch.linalg.svd(F, full_matrices=True)
                self.ssm.dynamics_mod.mean_fn.weight.data = (U * S.clip(max=1.0)) @ VmT

    def on_validation_epoch_end(self):
        # if self.current_epoch == 0 or self.current_epoch == (self.cfg.check_val_every_n_epoch - 1):
        #     self.log("r2_valid_enc", -1., on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #     self.log("r2_valid_prd", -1., on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #
        # elif self.current_epoch > 1 and self.current_epoch % 50 == 0:
        # n_time_bins_enc = self.n_time_bins_enc
        # m_train = torch.cat(self.train_z, dim=1).mean(dim=0)
        # m_valid = torch.cat(self.valid_z, dim=1).mean(dim=0)
        # veloc_train = torch.cat(self.train_veloc, dim=0)
        # veloc_valid = torch.cat(self.valid_veloc, dim=0)
        # n_trials, n_time_bins, n_latents = m_train.shape
        # n_veloc = veloc_train.shape[-1]

        # try:
        if self.current_epoch > 0:
            n_time_bins_enc = self.n_time_bins_enc
            m_train = torch.cat(self.train_z, dim=1).mean(dim=0)
            n_train_trials = m_train.shape[0]

            # reduce number of regressor trials for training speed
            train_trial_dx_regress = torch.randperm(n_train_trials)[:n_train_trials//5]

            m_valid = torch.cat(self.valid_z, dim=1).mean(dim=0)
            veloc_train = torch.cat(self.train_veloc, dim=0)
            veloc_valid = torch.cat(self.valid_veloc, dim=0)
            n_trials, n_time_bins, n_latents = m_train.shape
            n_veloc = veloc_train.shape[-1]

            # clf = MLPRegressor(hidden_layer_sizes=(100,), alpha=1e-1)
            # clf.fit(m_train[train_trial_dx_regress, :n_time_bins_enc].reshape(-1, n_latents).cpu(),
            #         veloc_train[train_trial_dx_regress, :n_time_bins_enc].reshape(-1, n_veloc).cpu())
            # r2_valid_enc = clf.score(m_valid[:, :n_time_bins_enc].reshape(-1, n_latents).cpu(),
            #                          veloc_valid[:, :n_time_bins_enc].reshape(-1, n_veloc).cpu())
            # r2_valid_prd = clf.score(m_valid[:, n_time_bins_enc:].reshape(-1, n_latents).cpu(),
            #                          veloc_valid[:, n_time_bins_enc:].reshape(-1, n_veloc).cpu())

            clf = skorch.regressor.NeuralNetRegressor(torch.nn.Sequential(torch.nn.Linear(n_latents, 100,
                                                                                          device=self.cfg.device),
                                                                          torch.nn.SiLU(),
                                                                          torch.nn.Linear(100, n_veloc,
                                                                                          device=self.cfg.device)),
                                                      max_epochs=100, device=self.cfg.device,
                                                      optimizer=torch.optim.Adam,
                                                      optimizer__weight_decay=1e-3, optimizer__lr=1e-3, verbose=0)

            # clf.fit(m_train[train_trial_dx_regress, :n_time_bins_enc].reshape(-1, n_latents),
            #         veloc_train[train_trial_dx_regress, :n_time_bins_enc].reshape(-1, n_veloc))
            clf.fit(m_train[train_trial_dx_regress, :n_time_bins_enc],
                    veloc_train[train_trial_dx_regress, :n_time_bins_enc])


            r2_valid_enc = clf.score(m_valid[:, :n_time_bins_enc].reshape(-1, n_latents).detach().cpu(),
                                     veloc_valid[:, :n_time_bins_enc].reshape(-1, n_veloc).detach().cpu())
            r2_valid_prd = clf.score(m_valid[:, n_time_bins_enc:].reshape(-1, n_latents).detach().cpu(),
                                     veloc_valid[:, n_time_bins_enc:].reshape(-1, n_veloc).detach().cpu())

            if r2_valid_enc >= self.best_r2_enc:
                self.best_clf_enc = copy.deepcopy(clf)
                self.best_ssm_enc = copy.deepcopy(self.ssm)
                self.best_r2_enc = r2_valid_enc
            if r2_valid_prd >= self.best_r2_prd:
                self.best_clf_prd = copy.deepcopy(clf)
                self.best_ssm_prd = copy.deepcopy(self.ssm)
                self.best_r2_prd = r2_valid_prd

        # except:
        else:
            r2_valid_enc = -1.
            r2_valid_prd = -1.

        self.log("r2_valid_enc", r2_valid_enc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("r2_valid_prd", r2_valid_prd, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.valid_z = []
        self.train_z = []
        self.valid_veloc = []
        self.train_veloc = []
