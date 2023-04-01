import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F


class SimCLR(pl.LightningModule):
    """
        This class implements the SimCLR model for any encoder and projection head
    """
    def __init__(self, aug_module, encoder, projector, optim_hparams, temperature):
        super().__init__()
        self.save_hyperparameters("aug_module", "encoder", "projector", "optim_hparams", "temperature")
        self.temperature = temperature
        assert self.temperature > 0.0, 'The temperature must be a positive float!'
        self.optim_hparams = optim_hparams
        self.f = encoder
        self.g = projector
        self.net = nn.Sequential(self.f, self.g)
        self.aug_module = aug_module

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.optim_hparams["lr"],
                                weight_decay=self.optim_hparams["weight_decay"])
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.optim_hparams['max_epochs'],
                                                            eta_min=self.optim_hparams['lr']/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        inputs, _ = batch  # Every input is a list of two augmented eeg epochs, make sure to use
        # ContrastiveTransformations!
        # inputs = torch.cat(inputs, dim=0)  # Put the first version of the augmented epochs together and then the
        # second version
        inputs = torch.squeeze(inputs, dim=1)
        with torch.no_grad():
            inputs = torch.cat((self.aug_module(inputs.clone()), self.aug_module(inputs.clone())))


        # Encode all sequences
        feats = self.net(inputs.unsqueeze(dim=1))  # Remove redundant dimension (which is not used in 1D convolutional network)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='test')


