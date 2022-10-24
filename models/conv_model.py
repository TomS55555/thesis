"""
    This module implements a simple 1D convolutional network for the processing of EEG-signals in the time domain

    The optimization part of this code is copied from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
"""

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim as optim
import constants


class CNN_block(nn.Module):
    """
        One CNN block consists of a 1D (3) convolution, a Max pooling and a Batch normalization
    """
    def __init__(self, input_size, kernel_size, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.net(x)


def create_model(model_hparams):
    model = nn.Sequential(
        CNN_block(constants.SLEEP_EPOCH_SIZE, 3, 1, 32),
        CNN_block(1500, 3, 32, 64),
        CNN_block(750, 3, 64, 64),
        nn.Flatten(),
        nn.Linear(in_features=375*64, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=10)
    )
    return model


class CNNmodel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CNNmodel")
        parser.add_argument("--model_hparams", nargs="*")
        parser.add_argument("--optimizer_name", type=str, default="Adam")
        parser.add_argument("--optimizer_hparams", nargs="*")
        # parser.add_argument("--hidden_layers", nargs=3, type=int, default=[256])
        return parent_parser

    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, **kwargs):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.loss_module = nn.CrossEntropyLoss()

        # Create model
        self.model = create_model(model_hparams)
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
        assert inputs.shape[1] == 1
        print("BATCH SHAPE: ", inputs.shape)
        preds = self.model(torch.squeeze(inputs, dim=1))  # Remove the epoch dimension of size 1
        loss = self.loss_module(preds, labels.squeeze().long())
        acc = (preds.argmax(dim=-1) == labels.squeeze()).float().mean()

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

