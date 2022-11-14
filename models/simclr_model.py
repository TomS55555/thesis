import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim
import constants
from models.mymodules import CNN_head, CNN_block, SimpleMLP
import torch.nn.functional as F


# TODO: Make this class generic for other heads
class CNNmodel_SimCLR(pl.LightningModule):
    """
        This class implements SimCLR with a 1D convolutional head
    """
    def __init__(self, cnn_encoder_hparams, projection_head_hparams, optim_hparams, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        self.optim_hparams = optim_hparams
        # Base model f(.)
        self.f = CNN_head(**cnn_encoder_hparams)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.g = SimpleMLP(
            in_features=cnn_encoder_hparams['representation_dim'],
            hidden_dim=4*projection_head_hparams['hidden_dim'],
            out_features=projection_head_hparams['hidden_dim']
        )
        self.net = nn.Sequential(self.f, self.g)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                **self.optim_hparams)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.optim_hparams['lr']/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        inputs, _ = batch
        inputs = torch.cat(inputs, dim=0)

        # Encode all sequences
        feats = self.net(torch.squeeze(inputs, dim=1))  # Remove redundant dimension (which is not used in 1D convolutional network)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
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


