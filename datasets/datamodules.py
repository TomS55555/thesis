from datasets.augmentations import *
from utils.helper_functions import prepare_data_features
import pytorch_lightning as pl
import torch.utils.data as data
import numpy as np
from datasets.datasets import SHHSdataset, SHHS_dataset_2
from utils.helper_functions import memReport, cpuStats


class EEGdataModule(pl.LightningDataModule):
    """
        This class acts as a wrapper for the SHHSdataset classes
        DO NOT FORGET TO CALL the setup method!!
    """

    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 data_split,
                 num_patients: int,
                 num_workers: int,
                 first_patient: int = 1,
                 num_ds: int = 1,  # This property can be used to load a different dataset every epoch
                 transform=None,
                 test_dl=None, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.first_patient = first_patient
        self.num_patients = num_patients
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_dl = test_dl
        self.num_ds = num_ds
        self.data_split = data_split

        self.num_patients_per_ds = num_patients // self.num_ds

        self.load_dataset(0)

    def load_dataset(self, idx):
        first_patient = self.first_patient + idx * self.num_patients_per_ds  # ! Make sure idx starts at 0
        eeg_trainval = SHHSdataset(data_path=self.data_path,
                                   first_patient=first_patient,
                                   num_patients=self.num_patients_per_ds)

        num = np.array(self.data_split).sum()
        piece = eeg_trainval.__len__() // num
        split = [self.data_split[0] * piece, eeg_trainval.__len__() - self.data_split[0] * piece]
        # print("BEFORE loading reassigning")
        # memReport()
        # cpuStats()
        self.eeg_train, self.eeg_val = data.random_split(eeg_trainval, split)
        # self.eeg_train = eeg_trainval
        # print("AFTER loading reassigning")
        # memReport()
        # cpuStats()
        # print("------------------------")

    def train_dataloader(self):
        print("CURRENT TRAINER EPOCH: ", self.trainer.current_epoch)
        if self.num_ds > 1:
            idx = self.trainer.current_epoch % self.num_ds  # self.trainer.current_epoch % self.num_ds
            self.load_dataset(idx)
        # TODO: set shuffle to true!
        return data.DataLoader(self.eeg_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                               drop_last=True, pin_memory=True)

    def val_dataloader(self):
        # TODO: change eeg_train beneath back into eeg_val!
        return data.DataLoader(self.eeg_val, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                               drop_last=True, pin_memory=True)

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
