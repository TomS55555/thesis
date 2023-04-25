import torch
import torch.nn as nn
from torch import optim

import constants
from models.sleep_transformer import OuterTransformer, Aggregator
import pytorch_lightning as pl

FEAT_DIM = 184


class CnnTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            CnnEncoder(),
            OuterTransformer(
                outer_dim=32,
                feat_dim=FEAT_DIM,
                dim_feedforward=1024,
                num_heads=8,
                num_layers=4
            ),
            Aggregator(feat_dim=FEAT_DIM),
            nn.Linear(FEAT_DIM, constants.N_CLASSES)
        )
        self.loss_module = nn.CrossEntropyLoss()
        self.save_hyperparameters()


    def common_step(self, batch, calculate_loss=False):
        inputs, labels = batch
        preds = self.net(inputs)

        loss = self.loss_module(preds, labels.long()) if calculate_loss else None

        acc = (preds.argmax(dim=-1) == labels).float().mean()

        return acc, loss


    def training_step(self, batch, batch_idx):
        acc, loss = self.common_step(batch, calculate_loss=True)

        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        acc, loss = self.common_step(batch, calculate_loss=True)
        self.log('val_acc', acc)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        acc, _ = self.common_step(batch, calculate_loss=False)

        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=1e-4,
            eps=1e-7,
            betas=(0.9, 0.999)
        )
        return optimizer


class CnnEncoder(nn.Module):
    """
        This cnn encoder takes as input vectors of size [batch x 1 x input_size] and outputs [batch x 32 x 188]
        If input_size == 12000: apply 6 convolutional filters
        else: apply 4
    """
    def __init__(self, input_size):
        super().__init__()
        if input_size == 4 * constants.SLEEP_EPOCH_SIZE:
            first_enc = nn.Sequential(
                nn.Conv1d(in_channels=1,
                          out_channels=64,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False),  # Output: 6000
                nn.GELU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(in_channels=64,
                          out_channels=64,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False),  # Output: 3000
                nn.GELU(),
                nn.BatchNorm1d(64)
            )
        elif input_size == constants.SLEEP_EPOCH_SIZE:
            first_enc = nn.Identity()
        else:
            print("No encoder supported for this epoch size, exiting...")
            exit(1)
        self.net = nn.Sequential(
            first_enc,
            nn.Conv1d(in_channels=64 if input_size != constants.SLEEP_EPOCH_SIZE else 1,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),  # Output: 1500
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),  # Output: 750
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),  # Output: 375
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=0,
                      dilation=2,
                      bias=False),  # Output: 184
            nn.BatchNorm1d(32)
        )

    def forward(self, x):
        return self.net(x)
