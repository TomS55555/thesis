"""
    This module implements a simple 1D convolutional network for the processing of EEG-signals in the time domain

    The optimization part of this code is copied from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
"""

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim
import constants
from models.mymodules import CNN_head, CNN_block, SimpleMLP
import torch.nn.functional as F


class CNNmodel_supervised(pl.LightningModule):

    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, **kwargs):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.loss_module = nn.CrossEntropyLoss()

        # Create model
        self.model = nn.Sequential(
            CNN_head(**model_hparams),
            SimpleMLP(in_features=int(constants.SLEEP_EPOCH_SIZE/8 * model_hparams["conv_filters"][-1]),
                      hidden_dim=model_hparams["hidden_dim"],
                      out_features=constants.N_CLASSES)
        )
        self.model_name = model_name

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 1, 3000), dtype=torch.float32)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(torch.squeeze(inputs, dim=1))  # Remove the epoch dimension of size 1
        acc = (preds.argmax(dim=-1) == labels.squeeze()).float().mean()

        loss = self.loss_module(preds, labels.squeeze().long())

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(torch.squeeze(inputs, dim=1))  # Remove the epoch dimension of size 1
        acc = (labels.squeeze() == preds.argmax(dim=-1)).float().mean()
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(torch.squeeze(inputs, dim=1))  # Remove the epoch dimension of size 1
        acc = (labels.squeeze() == preds.argmax(dim=-1)).float().mean()
        self.log('test_acc', acc)

