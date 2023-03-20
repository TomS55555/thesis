import os
import sys
sys.path.extend([os.getcwd()])

import time

import constants

import torch
import torch.nn as nn
from trainers.train_supervised import train_supervised
from argparse import Namespace

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print(device)
result_file_name = "transformer_simple"
train_path = "training"


args = {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": nn.TransformerEncoder(),
        "classifier": nn.Linear(
            in_features=100,
            out_features=constants.N_CLASSES
        ),

        "trainer_hparams": {
            "max_epochs": 40 * num_ds
            # "profiler": "simple"
        },
        "optim_hparams": {
            "lr": 1e-5,
            "weight_decay": 1e-3,
            "lr_hparams": None
        }
    }

if __name__ == "__main__":
    start = time.time()
    model, result = train_simclr(Namespace(**args), device)
    end = time.time()
    print(result)
    print("Elapsed time: ", (end-start))
