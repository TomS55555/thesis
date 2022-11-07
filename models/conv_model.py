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


class CNNmodel_SimCLR(pl.LightningModule):
    def __init__(self, model_name, model_hparams, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.f = CNN_head(**model_hparams)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.g = SimpleMLP(
            in_features=model_hparams['conv_filters'][-1],
            hidden_dim=4*hidden_dim,
            out_features=hidden_dim
        )
        self.net = nn.Sequential(self.f, self.g)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        inputs, _ = batch
        inputs = torch.cat(inputs, dim=0)

        # Encode all images
        feats = self.net(inputs)
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

