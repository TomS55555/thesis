import os
import sys

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.extend([os.getcwd()])

import time

import constants

import torch
import torch.nn as nn
from argparse import Namespace
from datasets.datamodules import EEGdataModule
from datasets.datasets import SHHS_dataset_STFT
import pytorch_lightning as pl
from models.sleep_transformer import SleepTransformer

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print(device)
result_file_name = "sleeptransformer01"
train_path = "training"
save_name = result_file_name
checkpoint_path = "checkpoints"

OUTER_DIM = 5
INNER_DIM = 29
FEAT_DIM = 128

data_args = {
        "data_path": constants.SHHS_PATH_GOOGLE,
        "data_split": [4, 1],
        "first_patient": 1,
        "num_patients": 500,
        "batch_size": 32,
        "num_workers": 2,
        "num_ds": 2,
        "exclude_test_set": constants.TEST_SET_BIG,
        "dataset_type": SHHS_dataset_STFT,
        "window_size": OUTER_DIM
    }


if __name__ == "__main__":
    start = time.time()
    dm = EEGdataModule(**data_args)
    model = SleepTransformer(
        outer_dim=OUTER_DIM,
        inner_dim=INNER_DIM,
        feat_dim=FEAT_DIM,
        dim_feedforward=1024,
        num_heads=8,
        num_layers=4
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, save_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,  # How many GPUs/CPUs to use
        reload_dataloaders_every_n_epochs=1 if data_args['num_ds'] > 1 else 0,
        # Reload dataloaders to get different part of the big dataset
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss",
                            save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
        max_epochs=50
    )
    trainer.fit(model, datamodule=dm)
    end = time.time()

    print("Elapsed time: ", (end-start))