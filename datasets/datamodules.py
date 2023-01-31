from datasets.augmentations import *
from utils.helper_functions import prepare_data_features
import pytorch_lightning as pl
import torch.utils.data as data
import numpy as np
from datasets.datasets import SHHSdataset


class EEGdataModule(pl.LightningDataModule):
    """
        This class acts as a wrapper for the SHHSdataset classes
        DO NOT FORGET TO CALL the setup method!!
    """

    def __init__(self, data_path, batch_size, data_split, num_patients, num_workers,
                 first_patient=1, transform=None, test_dl=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_dl = test_dl

        eeg_trainval = SHHSdataset(data_path=data_path,
                                   first_patient=first_patient,
                                   num_patients=num_patients,
                                   transform=transform)
        num = np.array(data_split).sum()
        piece = eeg_trainval.__len__() // num
        split = [data_split[0] * piece, eeg_trainval.__len__() - data_split[0] * piece]
        self.eeg_train, self.eeg_val = data.random_split(eeg_trainval, split)

    def train_dataloader(self):
        return data.DataLoader(self.eeg_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                               drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(self.eeg_val, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                               drop_last=True)

    def test_dataloader(self):
        if self.test_dl is None:
            raise NotImplementedError
        return self.test_dl


class SimCLRdataModule(pl.LightningDataModule):
    def __init__(self, pretrained_model, dm, batch_size, num_workers, device):
        super().__init__()
        self.train_ds = prepare_data_features(pretrained_model, dm.train_dataloader(), device)
        self.val_ds = prepare_data_features(pretrained_model, dm.val_dataloader(), device)
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataloader(self):
        return data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                               drop_last=False, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True,
                               drop_last=False, pin_memory=True, num_workers=self.num_workers)
