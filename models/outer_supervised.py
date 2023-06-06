import math

import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch import optim
from sklearn.metrics import cohen_kappa_score
from models.cnn_transformer import FEAT_DIM
import constants


class OuterSupervisedModel(pl.LightningModule):
    """
        General class for any combination of encoder head and classifier
    """

    def __init__(self,
                 encoder: nn.Module,
                 transformer: nn.Module,
                 classifier: nn.Module,
                 optim_hparams,
                 finetune_encoder=False,
                 finetune_transformer=False):
        super().__init__()
        self.save_hyperparameters()
          # encoder and classifier parameters are already saved because they are nn.Modules
        self.optim_hparams = optim_hparams
        for par in encoder.parameters():
            if finetune_encoder:
                par.requires_grad = True
            else:
                par.requires_grad = False
        for par in transformer.parameters():
            if finetune_transformer:
                par.requires_grad = True
            else:
                par.requires_grad = False
        self.encoder = encoder
        self.transformer = transformer
        self.classifier = classifier
        self.loss_module = nn.CrossEntropyLoss()


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
        inputs, labels = batch  # labels should be (batch x outer_labels)
        batch_dim, outer_dim, inner_dim, time_dim = inputs.shape  # should be (batch x outer x 1 x 3000)
        # reshape before input to encoder
        inputs_plus = inputs.view(batch_dim * outer_dim, inner_dim, time_dim)

        feats_plus = self.encoder(inputs_plus)  # output should be (batch*outer x feats)
        feats = feats_plus.view(batch_dim, outer_dim, FEAT_DIM)  # output (batch, outer, feats)
        # then put through transformer
        transformed_feats = self.transformer(feats)  # output should be still (batch x outer x feats)
        preds = self.classifier(transformed_feats)  # output should be (batch x outer x logits)

        labels_plus = labels.view(batch_dim * outer_dim, )
        preds_plus = preds.view(batch_dim * outer_dim, constants.N_CLASSES)
        loss = self.loss_module(preds_plus, labels_plus.long())

        acc = (preds_plus.argmax(dim=-1) == labels_plus).float().mean()

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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch  # labels should be (batch x outer_labels)
        batch_dim, outer_dim, inner_dim, time_dim = inputs.shape  # should be (batch x outer x 1 x 3000)
        # reshape before input to encoder
        inputs_plus = inputs.view(batch_dim * outer_dim, inner_dim, time_dim)
        feats_plus = self.encoder(inputs_plus)  # output should be (batch*outer x feats)
        feats = feats_plus.view(batch_dim, outer_dim, FEAT_DIM)  # output (batch, outer, feats)
        # then put through transformer
        transformed_feats = self.transformer(feats)  # output should be still (batch x outer x feats)
        preds = self.classifier(transformed_feats)  # output should be (batch x outer x logits)

        labels_plus = labels.view(batch_dim * outer_dim, )
        preds_plus = preds.view(batch_dim * outer_dim, constants.N_CLASSES)

        acc = (preds_plus.argmax(dim=-1) == labels_plus).float().mean()
        self.log('test_acc', acc)
        preds = preds_plus.argmax(dim=-1).cpu()
        labels = labels_plus.cpu()
        kappa = 1.0 if torch.equal(preds, labels) else cohen_kappa_score(preds, labels)
        self.log('kappa', kappa)
