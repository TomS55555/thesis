import pytorch_lightning as pl
import torch.utils.data as data
import h5py
import torch
import numpy as np


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
                pass
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
