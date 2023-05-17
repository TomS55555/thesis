import torch
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from models.cnn_transformer import FEAT_DIM


class MAEEG_Transformer(pl.LightningModule):
    """
        This class implements the SSL method, described in MAEEG: Masked autoencoder for EEG representation learning
        It works as follows: given a sequential set of n EEG-epochs, they are first sent through an encoder which may or
        may not update its parameters depending on the value of train_encoder to produce n feature vectors. A proportion
        of these vectors is then masked with a MASKED token and sent through the transformer. Finally a reconstruction head
        tries to reconstruct the whole sequence of n EEG-epochs and backpropagates the loss to pretrain the transformer.
    """

    def __init__(self, encoder: nn.Module, transformer: nn.Module, recon_head: nn.Module, optim_hparams,
                 train_encoder=False):
        super().__init__()
        self.save_hyperparameters()
        self.optim_hparams = optim_hparams
        self.encoder = encoder
        self.transformer = transformer
        self.recon_head = recon_head
        self.train_encoder = train_encoder

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
        masked_feats[torch.arange(b), mask_idxs, :] = torch.randn((feat,), device=feats.device)
        return masked_feats  # This returns a copy which is needed

    def reconstruction_loss(self, inputs, reconstructed_eeg):
        """
            This function calculates the reconstruction loss between inputs and reconstructed_inputs
            Inputs is a tensor of size (batch, outer, time)
            Reconstructed input is a tensor of size (batch, outer * 3000)

            The loss is calculated as 1 - dot(eeg, recon_eeg) / (norm(eeg) * norm(recon_eeg))
        """
        b, out, c, t = inputs.size()
        original_eeg = inputs.view(b, out*t)
        norm_eeg = torch.linalg.vector_norm(original_eeg, dim=-1)
        norm_recon_eeg = torch.linalg.vector_norm(reconstructed_eeg, dim=-1)
        loss = 1 - torch.linalg.vecdot(original_eeg, reconstructed_eeg, dim=-1) / (norm_eeg * norm_recon_eeg)
        return torch.mean(loss)

    def common_step(self, batch, mode):
        """
            Inputs should be a tensor of size (batch, outer, channel, time); e.g.: inputs.size() = (64, 4, 1, 3000)
        """
        inputs, _ = batch

        b, out, c, t = inputs.size()

        feats = self.encoder(inputs.view(b*out, c, t)).view(b, out, FEAT_DIM)  # feats of size (batch, outer, feat)
        with torch.no_grad():
            masked_feats = self.mask_feats(feats) # mask some of the feature vectors

        transformed_feats = self.transformer(masked_feats)  # output of size (batch, outer, feat)
        reconstructed_inputs = self.recon_head(transformed_feats)  # output of size (batch, 1, outer * 3000)

        loss = self.reconstruction_loss(inputs, reconstructed_inputs.squeeze(1))
        self.log(mode + "_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, mode="train")
