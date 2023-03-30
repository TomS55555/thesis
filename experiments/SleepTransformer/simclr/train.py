import os
import sys

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.extend([os.getcwd()])

import time

import constants
import math
import torch
import torch.nn as nn
from argparse import Namespace, ArgumentParser
from datasets.datamodules import EEGdataModule
from datasets.datasets import SHHS_dataset_STFT
import pytorch_lightning as pl
from models.sleep_transformer import InnerTransformer
from models.simclr_model import SimCLR
from utils.helper_functions import load_model
from datasets.augmentations import AugmentationModuleSTFT

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print(device)
save_name = "pretrained_inner_transformer"
train_path = "training"
checkpoint_path = "checkpoints"

OUTER_DIM = 1  # Only pretraining inner transformer
INNER_DIM = 29
FEAT_DIM = 128

HIDDEN_DIM = 256
Z_DIM = 128

NUM_PATIENTS = 100
PATIENTS_PER_DS = 250
NUM_DS = math.ceil(NUM_PATIENTS/PATIENTS_PER_DS)
MAX_EPOCHS = 2
TOTAL_EPOCHS = MAX_EPOCHS * NUM_DS

data_args = {
        "data_path": constants.SHHS_PATH_GOOGLE,
        "data_split": [4, 1],
        "first_patient": 1,
        "num_patients": NUM_PATIENTS,
        "batch_size": 256,
        "num_workers": 4,
        "num_ds": NUM_DS,
        "exclude_test_set": constants.TEST_SET_1,
        "dataset_type": SHHS_dataset_STFT,
        "window_size": OUTER_DIM
    }


def train(version: str):
    start = time.time()
    dm = EEGdataModule(**data_args)

    encoder = nn.Sequential(
        InnerTransformer(
            inner_dim=INNER_DIM,
            feat_dim=FEAT_DIM,
            dim_feedforward=1024,
            num_heads=8,
            num_layers=4
        ),
        nn.Flatten()  # this should result in a final layer of size outer x feat = feat bc outer=1 for pretraining
    )

    projection_head = nn.Sequential(
        nn.Linear(FEAT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, Z_DIM)
    )

    aug_module = AugmentationModuleSTFT(
        batch_size=data_args['batch_size'],
        time_mask_window=3,
        freq_mask_window=6
    )

    model = SimCLR(
        aug_module=aug_module,
        encoder=encoder,
        projector=projection_head,
        temperature=0.05,
        optim_hparams={
            "max_epochs": TOTAL_EPOCHS,
            "lr": 3e-4,
            "weight_decay": 1e-4
        }
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, save_name + version),
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
        max_epochs=TOTAL_EPOCHS
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None
    trainer.fit(model, datamodule=dm)
    end = time.time()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    train(args.version)
