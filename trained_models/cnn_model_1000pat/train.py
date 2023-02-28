import os
import sys
sys.path.extend([os.getcwd()])

import time

import constants

import torch
from trainers.train_simclr import train_simclr
from argparse import Namespace

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print(device)
result_file_name = "simclr5000pat"
train_path = "training"


args = {
  "MODEL_TYPE": "CNN_model_simclr",
  "save_name": result_file_name,
  "DATA_PATH": constants.SHHS_PATH_DEKSTOP,
  "CHECKPOINT_PATH": "checkpoints",
  "temperature": 0.05,

  "encoder": "CNN_head",
  "encoder_hparams": {
    "conv_filters": [32, 64, 64],
    "representation_dim": 100
  },

  "projection_head": "MLP",
  "projection_head_hparams": {
    "input_dim": 100,
    "hidden_dim": 300,
    "output_dim": 100
  },

  "aug_hparams": {
    "amplitude-min": 0.9,
    "amplitude-max": 1.1,
    "timeshift-min": 5,
    "timeshift-max": 300,
    "zeromask-min": 300,
    "zeromask-max": 500,
    "noise-min": 0,
    "noise-max": 0.3,
    "bandstop-min": 3,
    "bandstop-max": 45,
    "freq-window": 3
  },

  "data_hparams": {
    "first_patient": 1000,
    "num_patients": 50,
    "data_split": [4, 1],
    "batch_size": 512,
    "num_workers": 0,
    "exclude_test_set": constants.TEST_SET_1,
  },

  "trainer_hparams": {
    "max_epochs": 1,
    #"profiler": "pytorch"
  },

  "optim_hparams": {
    "max_epochs": 1,
    "lr": 3e-4,
    "weight_decay": 1e-4
  }
}

if __name__ == "__main__":
    start = time.time()
    model, result = train_simclr(Namespace(**args), device)
    end = time.time()
    print(result)
    print("Elapsed time: ", (end-start))
