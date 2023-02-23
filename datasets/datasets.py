import os.path
import constants
import torch.utils.data as data
import h5py
import torch
import numpy as np
from datasets.augmentations import ContrastiveTransformations


class SHHSdataset(torch.utils.data.Dataset):
    """
        This class loads the dataset as defined in /esat/biomeddata/SHHS_dataset/no_backup
        Only the *_eeg.mat files are used and the number of patients to use is specified in the initialization
        Some patients are missing in the files, but this is ignored
        Finally the data of the different patients are concatenated into one big tensor, which can then be indexed
        directly with __get_item__()
    """

    def __init__(self, data_path: str,
                 first_patient: int,
                 num_patients: int,
                 window_size: int = 1,
                 exclude_test_set: tuple = (),
                 transform: ContrastiveTransformations = None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.window_size = window_size
        X1_list = []
        labels_list = []
        for patient in range(first_patient, first_patient + num_patients):
            if patient in exclude_test_set:
                continue
            datapoint = data_path + "n" + f"{patient:0=4}" + "_eeg.mat"
            try:
                f = h5py.File(datapoint, 'r')
                x1 = torch.as_tensor(np.array(f.get("X1")))
                # Normalization
                DATA_MEANS = x1.mean(dim=0, keepdim=True)
                DATA_STD = x1.std(dim=0, keepdim=True)
                x1 = (x1 - DATA_MEANS) / DATA_STD
                x1 = x1[None, :]
                X1_list.append(x1.permute(2, 0, 1))
                label = torch.as_tensor(np.array(f.get("label"))[0])
                labels_list.append(label)
            except FileNotFoundError as e:
                print("Couldn't find file at path: ", datapoint)  # No problem if some patients are missing
        self.X1 = torch.cat(X1_list, 0)
        self.labels = torch.cat(labels_list, 0)
        self.labels = self.labels - torch.ones(self.labels.size(0))  # Change label range from 1->5 to 0->4s
        self.length = self.labels.size(0) - self.window_size  # Avoid problems at end of dataset
        if self.labels.size(0) == 0:
            raise FileNotFoundError  # No data found at all, raise an error

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        inputs = self.X1[item:item + self.window_size]  # Pointer to the data
        if self.transform is None:
            return inputs, self.labels[item:item + self.window_size]
        else:
            prev_inputs = self.X1[item - 1:item - 1 + self.window_size] if item > 0 else inputs
            return self.transform(inputs, prev_inputs), self.labels[item:item + self.window_size]


class SHHS_dataset_2(torch.utils.data.Dataset):
    """
        This class fetches data from the SHHS dataset by keeping a list of patients
        and an index which is a cumulative sum of all patients. The function __get_item__ can then directly
        index the correct file and fetch the correct datapoint.

        However this implementation turns out to be at least 10x slower than the other one and I will therefore
        discard it for now.
    """

    def __init__(self, data_path, first_patient, num_patients, window_size=1, transform=None):
        super().__init__()
        self.first_patient = first_patient
        self.num_patients = num_patients
        self.window_size = window_size
        self.data_path = data_path
        index_file_path = os.path.join(data_path, constants.PATIENT_INFO_FILE)
        if not os.path.exists(index_file_path):
            print("Could not find file: ", index_file_path)
            exit(1)
        with open(index_file_path, 'r') as f:
            index_file = f.readlines()
        self.paths = list()
        self.index = list()
        for line in index_file:
            path, idx = line.split('-')
            idx = int(idx)
            assert idx > 1
            self.paths.append(path)
            self.index.append(idx)
        self.index = np.cumsum(np.asarray(self.index))

    def __len__(self):
        if self.first_patient > 0:
            return (self.index[self.first_patient + self.num_patients] - self.index[
                self.first_patient - 1])
        return self.index[self.num_patients]

    def __getitem__(self, item):
        if self.first_patient > 0:
            index_item = item + self.index[self.first_patient - 1]
            patient_idx = np.argmax(self.index > index_item)
            item_in_patient = index_item - self.index[patient_idx - 1]
        else:
            index_item = item
            item_in_patient = item
            patient_idx = np.argmax(self.index > index_item)

        # datapoint = self.paths[patient_idx]
        datapoint = self.data_path + "/n" + f"{patient_idx + 1:0=4}" + "_eeg.mat"
        if not os.path.exists(datapoint):
            print("Couldn't find file: ", datapoint)
            exit(1)
        f = h5py.File(datapoint, 'r')
        X1 = torch.as_tensor(np.array(f.get("X1")))

        # Normalization
        DATA_MEANS = X1.mean(dim=0, keepdim=True)
        DATA_STD = X1.std(dim=0, keepdim=True)
        X1 = (X1 - DATA_MEANS) / DATA_STD

        labels = torch.as_tensor(np.array(f.get("label"))[0])

        if self.index[patient_idx] - index_item < self.window_size:  # Return last window of patient if not enough room
            # starting at item
            return X1[None, :, -self.window_size:].permute(2, 0, 1), labels[-self.window_size:]
        return X1[None, :, item_in_patient:item_in_patient + self.window_size].permute(2, 0, 1), \
               labels[item_in_patient:item_in_patient + self.window_size]
