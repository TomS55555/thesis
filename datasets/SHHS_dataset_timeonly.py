import pytorch_lightning as pl
import torch.utils.data as data
import h5py
import torch
import numpy as np


class EEGdataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser, **kwargs):
        parser = parent_parser.add_argument_group("CNNmodel")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_split", nargs="*", type=int, default=[3, 1, 1])
        parser.add_argument("--num_patients", type=int)
        return parent_parser

    def __init__(self, DATA_PATH, batch_size, data_split, num_patients, **kwargs):
        super().__init__()
        self.data_dir = DATA_PATH
        self.batch_size = batch_size
        self.data_split = data_split
        self.num_patients = num_patients

    def setup(self, stage=None):
        eeg_all = EEGdataset(self.data_dir, 1, 1+self.num_patients)
        num = np.array(self.data_split).sum()
        piece = eeg_all.__len__() // num
        split = [self.data_split[0] * piece, self.data_split[1] * piece, eeg_all.__len__() - (self.data_split[0] + self.data_split[1]) * piece]
        assert np.array(split).sum() == eeg_all.__len__()
        self.eeg_train, self.eeg_val, self.eeg_test = data.random_split(eeg_all, split)  # use a 3/5;1/5;1/5 split

    def train_dataloader(self):
        return data.DataLoader(self.eeg_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.eeg_val, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return data.DataLoader(self.eeg_test, batch_size=self.batch_size, shuffle=False)


class EEGdataset(torch.utils.data.Dataset):
    def __init__(self, data_path, first_patient=1, last_patient=10, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        X1_list = []
        labels_list = []
        for patient in range(first_patient, last_patient):
            datapoint = data_path + "n" + f"{patient:0=4}" + "_eeg.mat"
            try:
                f = h5py.File(datapoint, 'r')
                x1 = torch.Tensor(np.array(f.get("X1")))
                x1 = x1[None, :]
                X1_list.append(x1.permute(2, 0, 1))
                label = torch.Tensor(np.array(f.get("label"))[0])
                labels_list.append(label)
            except FileNotFoundError as e:
                print("Couldn't find file at path: ", datapoint)
        self.X1 = torch.cat(X1_list, 0)
        self.labels = torch.cat(labels_list, 0)
        self.labels = self.labels - torch.ones(self.labels.size(0))  # Change label range from 1->5 to 0->4
        if self.labels.size(0) == 0:
            raise FileNotFoundError     # Data not found

        # TODO: Normalization!!
        DATA_MEANS = self.X1.mean(dim=2, keepdim=True)
        DATA_STD = self.X1.std(dim=2, keepdim=True)
        self.X1 = (self.X1 - DATA_MEANS) / DATA_STD


    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, item):
        return self.X1[item], self.labels[item]

