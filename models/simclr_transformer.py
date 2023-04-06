import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F


class SimCLR_Transformer(pl.LightningModule):
    """
        This class implements the SimCLR model for any encoder and projection head
    """

    def __init__(self, aug_module, encoder, cont_projector, recon_projector, optim_hparams, alpha, temperature):
        super().__init__()
        self.save_hyperparameters()
        self.temperature = temperature
        assert self.temperature > 0.0, 'The temperature must be a positive float!'
        self.optim_hparams = optim_hparams
        self.f = encoder
        self.cont_projector = cont_projector
        self.recon_projector = recon_projector
        self.aug_module = aug_module
        self.alpha = alpha  # This parameter determines the relative weight of each loss
        self.recon_loss = nn.L1Loss()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.optim_hparams["lr"],
                                weight_decay=self.optim_hparams["weight_decay"])
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.optim_hparams['max_epochs'],
                                                            eta_min=self.optim_hparams['lr'] / 50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, feats, mode='train'):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        # Logging loss
        #self.log(mode + '_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:, None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + '_acc_top1', (sim_argsort == 0).float().mean())
        #print("Accuracy: ", (sim_argsort == 0).float().mean())
        self.log(mode + '_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode + '_acc_mean_pos', 1 + sim_argsort.float().mean())

        return nll

    def total_loss(self, batch, mode):
        inputs, _ = batch  # Every input is a list of two augmented eeg epochs, make sure to use
        # ContrastiveTransformations!
        # inputs = torch.cat(inputs, dim=0)  # Put the first version of the augmented epochs together and then the
        # second version

        inputs = torch.squeeze(inputs, dim=1)
        with torch.no_grad():
            augmented_inputs = torch.cat((self.aug_module(inputs.clone()), self.aug_module(inputs.clone()))).unsqueeze(
                dim=1)

        encoded_augmented_inputs = self.f(augmented_inputs)

        contrastive_outputs = self.cont_projector(encoded_augmented_inputs)
        reconstructed_outputs = self.recon_projector(encoded_augmented_inputs)

        contrastive_loss = self.info_nce_loss(contrastive_outputs, mode)
        reconstruction_loss = self.recon_loss(augmented_inputs.squeeze(dim=1), reconstructed_outputs)

        loss = self.alpha * reconstruction_loss + contrastive_loss
        self.log(mode + "_recon_loss", reconstruction_loss)
        self.log(mode + "_contrastive_loss", contrastive_loss)
        self.log(mode + "_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.total_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.total_loss(batch, mode='val')

