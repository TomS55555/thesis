import math

import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch import optim
from sklearn.metrics import cohen_kappa_score


class SupervisedModel(pl.LightningModule):
    """
        General class for any combination of encoder head and classifier
    """

    def __init__(self,
                 encoder: nn.Module,
                 classifier: nn.Module,
                 optim_hparams):
        super().__init__()
          # encoder and classifier parameters are already saved because they are nn.Modules
        self.optim_hparams = optim_hparams
        self.encoder = encoder
        self.classifier = classifier
        self.net = nn.Sequential(encoder, classifier)
        self.loss_module = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.optim_hparams['lr'],
                                weight_decay=self.optim_hparams['weight_decay'])

        if self.optim_hparams['lr_hparams'] is not None:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, **self.optim_hparams['lr_hparams'])
            return [optimizer], [scheduler]
        else:
            return optimizer

    def common_step(self, batch, calculate_loss=False):
        inputs, labels = batch
        preds = self.net(inputs)  # Remove the epoch dimension of size 1
        acc = (preds.argmax(dim=-1) == labels.squeeze()).float().mean()

        loss = self.loss_module(preds, labels.squeeze(dim=-1).long()) if calculate_loss else None

        return acc, loss

    def training_step(self, batch, batch_idx):
        # print("Batch idx: ", batch_idx, "-----------")
        # print("Labels: ", batch[1])
        acc, loss = self.common_step(batch, calculate_loss=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        acc, loss = self.common_step(batch, calculate_loss=True)
        self.log('val_acc', acc)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch


        preds = self.net(torch.squeeze(inputs, dim=1))  # Remove the epoch dimension of size 1
        preds = preds.argmax(dim=-1)
        labels = labels.squeeze(dim=-1).type(torch.int64)
        acc = (preds == labels).float().mean()

        self.log('test_acc', acc)
        preds.cpu()
        labels.cpu()
        kappa = 1.0 if torch.equal(preds, labels) else cohen_kappa_score(preds, labels)
        self.log('kappa', kappa)
