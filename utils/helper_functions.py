from datasets.SHHS_dataset_timeonly import EEGdataModule
from datasets.augmentations import ContrastiveTransformations, AmplitudeScale, TimeShift, ZeroMask, GaussianNoise, \
    BandStopFilter, DCShift

import os
import torch
from copy import deepcopy
import torch.nn as nn
import torch.utils.data as data
from tqdm.notebook import tqdm
import pytorch_lightning as pl


class SimCLRdataModule(pl.LightningDataModule):
  def __init__(self, pretrained_model, dm, batch_size, num_workers, device):
      super().__init__()
      self.train_ds = prepare_data_features(pretrained_model, dm.train_dataloader(), device)
      self.val_ds = prepare_data_features(pretrained_model, dm.val_dataloader(), device)
      self.test_ds = prepare_data_features(pretrained_model, dm.test_dataloader(), device)
      self.num_workers = num_workers
      self.batch_size = batch_size

  def train_dataloader(self):
    return data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=self.num_workers)
  def val_dataloader(self):
    return data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=self.num_workers)
  def test_dataloader(self):
    return data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                                   drop_last=False, pin_memory=True, num_workers=self.num_workers)

@torch.no_grad()
def prepare_data_features(model, data_loader, device):
    """
        TODO: check whether deepcopy actually works when you want to train the encoder as well
    """
    # Prepare model
    network = deepcopy(model.f)
    network.g = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    feats, labels = [], []
    for batch_inputs, batch_labels in tqdm(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_feats = network(batch_inputs.squeeze(dim=1))
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    return data.TensorDataset(feats, labels)


def load_model(model_type, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        model = model_type.load_from_checkpoint(
            checkpoint_path)  # Automatically loads the model with the saved hyperparameters
    else:
        print("Model at location ", checkpoint_path, " not found!")
        exit(1)
    return model


def prepare_data_module(data_path, **data_hparams):
    p = data_hparams["transform-prob"]  # probability of applying the transforms
    contrast_transforms = ContrastiveTransformations(
        [
            AmplitudeScale(data_hparams["amplitude-min"], data_hparams["amplitude-max"], p, 1),
            GaussianNoise(data_hparams["noise-min"], data_hparams["noise-max"], p, 1),
            ZeroMask(data_hparams["zeromask-min"], data_hparams["zeromask-max"], p, 1),
            TimeShift(data_hparams["timeshift-min"], data_hparams["timeshift-max"], p, 1),
            BandStopFilter(data_hparams['bandstop-min'], data_hparams["bandstop-max"], p, 1,
                           data_hparams['freq-window'])
        ], n_views=2
    )
    data_module = EEGdataModule(
        DATA_PATH=data_path,
        transform=contrast_transforms,
        **data_hparams)
    data_module.setup()
    return data_module
