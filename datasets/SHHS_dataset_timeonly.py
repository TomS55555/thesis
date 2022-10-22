import pytorch_lightning as pl
import torch.utils.data as data
import h5py
import torch
import numpy as np


"""
    This class acts as a wrapper for the EEGdataset class (also in this file)
    DO NOT FORGET TO CALL the setup method!!  
"""
class EEGdataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser, **kwargs):
        parser = parent_parser.add_argument_group("EEGdataModule")
        parser.add_argument("--DATA_PATH", type=str)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_split", nargs="*", type=int, default=[3, 1])
        parser.add_argument("--num_patients_train", type=int)
        parser.add_argument("--num_patients_test", type=int)
        return parent_parser

    def __init__(self, DATA_PATH, batch_size, data_split, num_patients_train, num_patients_test, num_workers, first_patient=1, **kwargs):
        super().__init__()
        self.data_dir = DATA_PATH
        self.batch_size = batch_size
        self.data_split = data_split
        self.num_patients_train = num_patients_train
        self.num_patients_test = num_patients_test
        self.first_patient_train = first_patient
        self.first_patient_test = self.first_patient_train+self.num_patients_train
        self.num_workers = num_workers

    def setup(self, stage=None):
        #TODO: maybe add this code to the initialization?
        eeg_trainval = EEGdataset(data_path=self.data_dir,
                                  first_patient=self.first_patient_train,
                                  num_patients=self.num_patients_train)
        num = np.array(self.data_split).sum()
        piece = eeg_trainval.__len__() // num
        split = [self.data_split[0] * piece, eeg_trainval.__len__() - self.data_split[0] * piece]

        assert np.array(split).sum() == eeg_trainval.__len__()

        self.eeg_train, self.eeg_val = data.random_split(eeg_trainval, split)

        self.eeg_test = EEGdataset(data_path=self.data_dir,
                                   first_patient=self.first_patient_test,
                                   num_patients=self.num_patients_test)

    def train_dataloader(self):
        return data.DataLoader(self.eeg_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(self.eeg_val, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(self.eeg_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


"""
    This class loads the dataset as defined in /esat/biomeddata/SHHS_dataset/no_backup
    Only the *_eeg.mat files are used and the number of patients to use is specified in the initialization
    Some patients are missing in the files, but this is ignored
    Finally the data of the different patients are concatenated into one big tensor
    
    I am aware that this implementation might not be computationally optimal: it is probably better to load a patient from a file,
    normalize the data and concatenate it immediatly into a tensor
"""
class EEGdataset(torch.utils.data.Dataset):
    def __init__(self, data_path, first_patient, num_patients, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        X1_list = []
        labels_list = []
        # TODO: Make the fetching of data more efficient
        for patient in range(first_patient, first_patient+num_patients):
            datapoint = data_path + "n" + f"{patient:0=4}" + "_eeg.mat"
            try:
                f = h5py.File(datapoint, 'r')
                x1 = torch.Tensor(np.array(f.get("X1")))
                x1 = x1[None, :]
                X1_list.append(x1.permute(2, 0, 1))
                label = torch.Tensor(np.array(f.get("label"))[0])
                labels_list.append(label)
            except FileNotFoundError as e:
                print("Couldn't find file at path: ", datapoint)  # No problem if some patients are missing
        self.X1 = torch.cat(X1_list, 0)
        self.labels = torch.cat(labels_list, 0)
        self.labels = self.labels - torch.ones(self.labels.size(0))  # Change label range from 1->5 to 0->4
        if self.labels.size(0) == 0:
            raise FileNotFoundError     # No data found at all, raise an error

        # Normalization
        DATA_MEANS = self.X1.mean(dim=2, keepdim=True)
        DATA_STD = self.X1.std(dim=2, keepdim=True)
        self.X1 = (self.X1 - DATA_MEANS) / DATA_STD

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, item):
        return self.X1[item], self.labels[item]

