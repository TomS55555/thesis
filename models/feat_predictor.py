import torch
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from models.cnn_transformer import FEAT_DIM
from models.sleep_transformer import Aggregator


class FeatPredictor(pl.LightningModule):
    """
        This class implements a new SSL method:
        It works as follows: given a sequential set of n EEG-epochs, they are first sent through an encoder which may or
        may not update its parameters depending on the value of train_encoder to produce n feature vectors. A proportion
        of these vectors is then masked with a MASKED token and sent through the transformer. The unmasked vectors are also
         sent through the encoder, but with a stop-gradient operation.
         Finally a projection head maps the transformer output of the masked features to the transformer output of the
         unmasked features and a reconstruction loss is backpropagated.
    """

    def __init__(self, encoder: nn.Module, transformer: nn.Module, proj_head: nn.Module, optim_hparams,
                 train_encoder=False):
        super().__init__()
        self.save_hyperparameters()
        self.optim_hparams = optim_hparams
        self.encoder = encoder
        self.transformer = transformer
        self.proj_head = proj_head
        self.train_encoder = train_encoder
        self.aggregator = Aggregator(feat_dim=FEAT_DIM)

        if self.train_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.optim_hparams["lr"],
                                weight_decay=self.optim_hparams["weight_decay"])
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.optim_hparams['max_epochs'],
                                                            eta_min=self.optim_hparams['lr'] / 50)
        return [optimizer], [lr_scheduler]

    def mask_feats(self, feats):
        """
            feats should be a tensor of size (batch x outer x feat)
            This function returns a copy such that the original is not affected and can be used for reconstruction
        """
        b, outer, feat = feats.size()
        mask_idxs = torch.randint(0, outer, (b,))  # random idx to mask for each example of the batch
        mask_token = -100 # Some value outside the range of the input EEG features
        masked_feats = feats.clone()
        what_is_masked = masked_feats[torch.arange(b), mask_idxs, :].clone()
        masked_feats[torch.arange(b), mask_idxs, :] = torch.randn((feat,), device=feats.device)
        return masked_feats, what_is_masked  # This returns a copy which is needed


    def reconstruction_loss(self, original, projected):
        """
            This function calculates the reconstruction loss between original and projected
            Both are tensors of size (batch, feat)

            The loss is calculated as the l2 normalized MSE
        """
        norm_original = torch.linalg.vector_norm(original, dim=-1)
        norm_projected = torch.linalg.vector_norm(projected, dim=-1)
        loss = 1 - torch.linalg.vecdot(original, projected, dim=-1) / (norm_original * norm_projected)
        return torch.mean(loss)


    def common_step(self, batch, mode):
        """
            Inputs should be a tensor of size (batch, outer, channel, time); e.g.: inputs.size() = (64, 4, 1, 3000)
        """
        inputs, _ = batch

        b, out, c, t = inputs.size()

        # First encode the set of 'outer' epochs into a set of 'outer' features
        feats = self.encoder(inputs.view(b*out, c, t)).view(b, out, FEAT_DIM)  # feats of size (batch, outer, feat)
        with torch.no_grad():
            masked_feats, what_is_masked = self.mask_feats(feats)  # mask some of the feature vectors

        # Send masked feats through transformer and then through projection head
        transformed_feats_masked = self.transformer(masked_feats)  # output of size (batch, outer, feat)
        aggregated_feats_masked = self.aggregator(transformed_feats_masked)
        projected_feats = self.proj_head(aggregated_feats_masked)  # output of size (batch, feat)

        loss = self.reconstruction_loss(what_is_masked, projected_feats)
        self.log(mode + "_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, mode="val")