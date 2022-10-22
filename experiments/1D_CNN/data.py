import pytorch_lightning as pl
from datasets.SHHS_dataset_timeonly import EEGdataset
import torch.utils.data as data


class EEGdataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        eeg_all = EEGdataset(self.data_dir, 1, 11)
        piece = eeg_all.__len__()//5
        self.eeg_train, self.eeg_val, self.eeg_test = data.random_split(eeg_all, [3*piece, piece, eeg_all.__len__()-4*piece])  # use a 3/5;1/5;1/5 split

    def train_dataloader(self):
        return data.DataLoader(self.eeg_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.eeg_val, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return data.DataLoader(self.eeg_test, batch_size=self.batch_size, shuffle=False)