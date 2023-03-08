from datasets.augmentations import ContrastiveTransformations, AmplitudeScale, TimeShift, ZeroMask, GaussianNoise, \
    BandStopFilter, DCShift

import os
import torch
from copy import deepcopy
import torch.nn as nn
import torch.utils.data as data
from tqdm.notebook import tqdm
import gc
import sys
import psutil


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if len(obj.size()) == 3:
                print("TENSOR: ", type(obj), obj.size())

def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

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


def get_checkpoint_path(train_path, save_name):
    """
        This function finds the trained model from a given checkpoint path and save name
    """
    rest_path = 'lightning_logs/version_0/checkpoints'
    dir_path = os.path.join(train_path, save_name, rest_path)
    dirs = os.listdir(dir_path)
    ckpt = list(filter(lambda x: x.startswith("epoch"), dirs))[0]
    print("Found checkpoint: ", ckpt)
    return os.path.join(dir_path, ckpt)



