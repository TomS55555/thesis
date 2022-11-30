import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch import optim


class SupervisedModel(pl.LightningModule):
    """
        General class for any combination of encoder head and classifier
    """

    def __init__(self, encoder, classifier, optim_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.optim_hparams = optim_hparams
        self.net = nn.Sequential(encoder, classifier)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.optim_hparams)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def common_step(self, batch, calculate_loss=False):
        inputs, labels = batch
        preds = self.net(torch.squeeze(inputs, dim=1))  # Remove the epoch dimension of size 1
        acc = (preds.argmax(dim=-1) == labels.squeeze()).float().mean()

        loss = self.loss_module(preds, labels.squeeze().long()) if calculate_loss else None

        return acc, loss

    def training_step(self, batch, batch_idx):
        acc, loss = self.common_step(batch, calculate_loss=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        acc, _ = self.common_step(batch)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        acc, _ = self.common_step(batch)
        self.log('test_acc', acc)