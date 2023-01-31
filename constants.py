import torch.nn as nn
from models.mymodules import CNN_head, SimpleMLP

SLEEP_EPOCH_SIZE = 3000
PATIENT_INFO_FILE = 'patient_mat_list.txt'
N_CLASSES = 5
fs = 100  # Sampling frequency

SHHS_PATH_ESAT = "/esat/biomeddata/SHHS_Dataset/no_backup/"
SHHS_PATH_DEKSTOP = "C:/Users/tomsm/PycharmProjects/thesis/data/"

"""
    These are the implemented encoders and projection heads which can be used in the json files
"""
ENCODERS = {
    "CNN_head": CNN_head,
    "None": nn.Identity
}

PROJECTION_HEADS = {
    "MLP": SimpleMLP
}

CLASSIFIERS = {
    "logistic": lambda input_dim: nn.Linear(input_dim, N_CLASSES)
}
