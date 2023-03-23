import os
import sys
sys.path.extend([os.getcwd()])

import time

import constants

import torch
import torch.nn as nn
from trainers.train_supervised import train_supervised
from argparse import Namespace
from datasets.datamodules import EEGdataModule
import pytorch_lightning as pl
from models.supervised_model import SupervisedModel

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print(device)
result_file_name = "transformer_simple"
train_path = "training"
save_name = "test_transformer"
checkpoint_path = "checkpoints"


args = {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": nn.TransformerEncoderLayer(d_model=3000, nhead=8, batch_first=True),
        "classifier": nn.Linear(
            in_features=100,
            out_features=constants.N_CLASSES
        ),

        "trainer_hparams": {
            "max_epochs": 1
            # "profiler": "simple"
        },
        "optim_hparams": {
            "lr": 1e-5,
            "weight_decay": 1e-3,
            "lr_hparams": None
        }
    }

data_args = {
        "data_path": constants.SHHS_PATH_DEKSTOP,
        "data_split": [4, 1],
        "first_patient": 1,
        "num_patients": 10,
        "batch_size": 64,
        "num_workers": 0,
        "num_ds": 1,
        "exclude_test_set": constants.TEST_SET_1
    }
if __name__ == "__main__":
    start = time.time()
    dm = EEGdataModule(**data_args)
    model = SupervisedModel(
        encoder=nn.TransformerEncoderLayer(d_model=3000, nhead=8, batch_first=True),
        classifier=nn.Linear(2048, 5),
        optim_hparams={

        }
    )
    trainer = pl.Trainer(

    )
    end = time.time()
    print(result)
    print("Elapsed time: ", (end-start))
