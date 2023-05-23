import torch
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from models.cnn_transformer import FEAT_DIM


class RandomShuffleTransformer(pl.LightningModule):
    """
        This class implements a new SSL method: Random Shuffle
        It works as follows: given a batch of a sequential set of n EEG-epochs, they are first sent through an encoder which may or
        may not update its parameters depending on the value of train_encoder to produce n feature vectors.

        A random index in the range(n) is chosen for each example of the batch. They are then shuffled and the pretext task
        for the network to solve is to find out which index is out of place.
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

        self.loss_module = nn.CrossEntropyLoss()

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

    def pretext_task(self, feats):
        """
            feats should be a tensor of size (batch x outer x feat)
            This function defines the following pretext task: in every example of the batch, one index in the range(outer)
            is chosen. Thus there are a number of 'batch' indices, corresponding to a number of 'batch' feature vectors.
            Those are shuffled throughout the batch and the pretext
            task is then for the network to predict which index of every example has been replaced.
            The function returns both the shuffled indices which are the speudolabels and the shuffled feats
        """
        b, outer, feat = feats.size()
        to_shuffle_idxs = torch.randint(0, outer, (b,)).to(feats).type(torch.long)  # random idx to shuffle for each example

        randperm = torch.randperm(b, dtype=torch.long).to(feats).type(torch.long)  # create a random permutation within the batch
        shuffled_feats = feats.clone()

        idxs = torch.arange(b, dtype=torch.long).to(feats).type(torch.long)
        shuffled_feats[idxs, to_shuffle_idxs, :] = feats[randperm, to_shuffle_idxs[randperm], :]  # old location = new location

        return to_shuffle_idxs, shuffled_feats


    def common_step(self, batch, mode):
        """
            Inputs should be a tensor of size (batch, outer, channel, time); e.g.: inputs.size() = (64, 4, 1, 3000)
        """
        inputs, _ = batch

        b, out, c, t = inputs.size()

        # First encode the set of 'outer' epochs into a set of 'outer' features
        feats = self.encoder(inputs.view(b*out, c, t)).view(b, out, FEAT_DIM)  # feats of size (batch, outer, feat)
        # create pretext task
        with torch.no_grad():
            pseudo_labels, shuffled_feats = self.pretext_task(feats)

        # Send shuffled feats through transformer and then through projection head
        transformed_feats = self.transformer(shuffled_feats)  # output of size (batch, outer, feat)
        preds = self.proj_head(transformed_feats)  # output of size (batch, outer), logits over class labels

        acc = (preds.argmax(dim=-1) == pseudo_labels.squeeze()).float().mean()

        loss = self.loss_module(preds, pseudo_labels.long())

        return acc, loss

    def training_step(self, batch, batch_idx):
        acc, loss = self.common_step(batch, mode="train")
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        acc, loss = self.common_step(batch, mode="val")
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return loss