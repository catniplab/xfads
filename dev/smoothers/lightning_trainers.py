import torch
import dev.utils as utils
import pytorch_lightning as lightning


class LightningNonlinearSSM(lightning.LightningModule):
    def __init__(self, ssm, cfg):
        super().__init__()

        self.ssm = ssm
        self.cfg = cfg
        self.n_samples = cfg.n_samples
        self.p_mask_y_in = cfg.p_mask_y_in
        self.p_mask_apb = cfg.p_mask_apb
        self.p_mask_a = cfg.p_mask_a
        self.p_mask_b = cfg.p_mask_b

        self.save_hyperparameters(ignore=['ssm', 'cfg'])

    def training_step(self, batch, batch_idx):
        y = batch[0]

        p_mask_a_t = self.p_mask_a
        p_mask_b_t = self.p_mask_b
        p_mask_apb_t = self.p_mask_apb
        p_mask_y_in_t = self.p_mask_y_in

        loss, z_s, stats = self.ssm(y, self.n_samples, p_mask_a=p_mask_a_t, p_mask_apb=p_mask_apb_t,
                                    p_mask_y_in=p_mask_y_in_t, p_mask_b=p_mask_b_t)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[0]

        loss, z_s, stats = self.ssm(y, self.n_samples)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ssm.parameters(), lr=self.cfg.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        log_Q_min = utils.softplus_inv(1e-2)
        log_Q_0_min = utils.softplus_inv(1e-1)

        self.ssm.dynamics_mod.log_Q.data = torch.clip(self.ssm.dynamics_mod.log_Q.data, min=log_Q_min)
        self.ssm.initial_c_pdf.log_Q_0.data = torch.clip(self.ssm.initial_c_pdf.log_Q_0.data, min=log_Q_0_min)

    # def on_epoch_end(self):
    #     if self.current_epoch % 10 == 0:
    #         torch.save(self.model.state_dict(), 'model.ckpt')

