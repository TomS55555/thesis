import math
import os
import sys
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.extend([os.getcwd()])
import pytorch_lightning as pl
import constants
import torch
import torch.nn as nn
from datasets.datasets import SHHSdataset
from datasets.datamodules import EEGdataModule
from models.cnn_transformer import CnnTransformer
from utils.helper_functions import get_data_path


train_path = "cnn_transformer_trainings"

OUTER_DIM = 4  # Use two previous and one following epoch for context

PATIENTS_PER_DS = 200  # Depends on RAM size of PC


def train_fully_supervised(device):
    num_patients = 1000
    max_epochs = 50

    dm = EEGdataModule(
        data_path=get_data_path(),
        data_split=[4, 1],
        first_patient=1,
        num_patients=num_patients,
        batch_size=64,
        num_workers=4,
        num_ds=math.ceil(num_patients / PATIENTS_PER_DS),
        exclude_test_set=constants.TEST_SET_1,
        dataset_type=SHHSdataset,
        window_size=OUTER_DIM
    )

    model = CnnTransformer()

    trainer = pl.Trainer(
        default_root_dir=train_path,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,  # How many GPUs/CPUs to use
        reload_dataloaders_every_n_epochs=1 if dm.num_ds > 1 else 0,
        # Reload dataloaders to get different part of the big dataset
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss",
                            save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
        max_epochs=max_epochs
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None
    trainer.fit(model, datamodule=dm)
    return model


def test(model, device):
    dm = EEGdataModule(
        data_path=get_data_path(),
        data_split=[4, 1],
        first_patient=1,
        num_patients=10,
        batch_size=64,
        num_workers=4,
        num_ds=1,
        exclude_test_set=constants.TEST_SET_1,
        test_set=True,
        dataset_type=SHHSdataset,
        window_size=OUTER_DIM
    )

    trainer = pl.Trainer(
        default_root_dir=train_path,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,  # How many GPUs/CPUs to use
        reload_dataloaders_every_n_epochs=1 if dm.num_ds > 1 else 0,
        # Reload dataloaders to get different part of the big dataset
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss",
                            save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
        max_epochs=10
    )

    result = trainer.test(model, dm)
    print(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(dev)

    if args.mode == "train":
        train_fully_supervised(dev)
    elif args.mode == "test":
        pass
    elif args.mode == "both":
        model = train_fully_supervised(dev)
        test(model, dev)
    else:
        print("Mode not recognized... Abort")
        exit(1)





