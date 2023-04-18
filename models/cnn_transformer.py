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
        batch_dim, outer_dim, feat_dim = inputs.shape
        preds = self.net(inputs)

        labels_plus = labels.view(batch_dim*outer_dim,)

        preds_plus = preds.view(batch_dim*outer_dim, constants.N_CLASSES)
        loss = self.loss_module(preds_plus, labels_plus.long()) if calculate_loss else None

        acc = (preds_plus.argmax(dim=-1) == labels_plus).float().mean()

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
        This cnn encoder takes as input vectors of size [batch x 1 x 12000] and outputs [batch x 32 x 188]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64,
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
