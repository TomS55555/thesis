import os
import sys

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.extend([os.getcwd()])

import constants
import math
import torch
import torch.nn as nn
from argparse import Namespace, ArgumentParser
from datasets.datamodules import EEGdataModule
from datasets.datasets import SHHSdataset
import pytorch_lightning as pl
from models.simclr_transformer import SimCLR_Transformer
from models.cnn_transformer import CnnEncoder, FEAT_DIM
from models.sleep_transformer import OuterTransformer, Aggregator
from utils.helper_functions import load_model, get_data_path
from datasets.augmentations import AugmentationModule
from trainers.train_simclr_classifiers import train_networks, test_networks
import json
from models.random_shuffle_transformer import RandomShuffleTransformer
from models.supervised_model import SupervisedModel
patients_list = [3, 5, 10, 20, 50, 100, 250]

OUTER_DIM = 6  # Only 1 and 4 are supported at the moment

PATIENTS_PER_DS = 250  # Depends on RAM size of PC

train_path = "randomshuffle_trainings"  # path used for training the networks
result_file_name = "test_results_randomshuffle_transformer"

pretrained_save_name = "pretrained_IT"

# parameters for SimCLR projection head
HIDDEN_DIM = 256
Z_DIM = 128


def get_encoder():
    """
        Input: batches of size [batch x outer x eeg-time]
        Output: batches of size [batch x outer x feat]
    """
    return nn.Sequential(
        CnnEncoder(constants.SLEEP_EPOCH_SIZE),
        Aggregator(feat_dim=FEAT_DIM)
    )


def get_transformer():
    """
        Input: batches of size [batch x outer x feat]
        Output: same size
    """
    return OuterTransformer(
        outer_dim=OUTER_DIM,
        feat_dim=FEAT_DIM,
        dim_feedforward=1024,
        num_heads=8,
        num_layers=4
    )


def get_projection_head():
    """
        The input to the projection head is the output of the transformer:
        [batch x outer x feat] and the output is of size [batch x outer] since it outputs the logits
        over the pseudolabels
    """
    return nn.Sequential(
        Aggregator(feat_dim=FEAT_DIM),
        nn.Linear(FEAT_DIM, HIDDEN_DIM),
        nn.GELU(),
        nn.Linear(HIDDEN_DIM, OUTER_DIM))


def get_classifier():
    return nn.Linear(
        FEAT_DIM,
        constants.N_CLASSES
    )


def get_data_args(num_patients, batch_size, num_workers=4):
    return {
        "data_path": get_data_path(),
        "data_split": [4, 1],
        "first_patient": 1,
        "num_patients": num_patients,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_ds": math.ceil(num_patients / PATIENTS_PER_DS),
        "exclude_test_set": constants.TEST_SET_BIG,
        "dataset_type": SHHSdataset,
        "window_size": OUTER_DIM
    }

def pretrain(device, version, patients, encoder=None):
    # TODO: fix normalization of STFT images!
    num_patients = patients
    batch_size = 64
    max_epochs = 220
    dm = EEGdataModule(**get_data_args(num_patients=num_patients,
                                       batch_size=batch_size))
    model = RandomShuffleTransformer(
        encoder=get_encoder() if encoder is None else encoder,
        transformer=get_transformer(),
        proj_head=get_projection_head(),
        optim_hparams={
            "max_epochs": max_epochs,
            "lr": 1e-4,
            "weight_decay": 1e-5
        },
        train_encoder=False
    )

    save_name = pretrained_save_name + '_' + str(num_patients) + 'pat'
    checkpoint_path = train_path + str(version)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, save_name + str(version)),
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




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--version", required=False, default="0")
    parser.add_argument("--pretrained_encoder_path", required=False, default=None)
    parser.add_argument("--num_patients", required=True, type=int)
    args = parser.parse_args()

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(dev)

    version = int(args.version)

    if args.mode == "pretrain":
        encoder = load_model(SupervisedModel, args.pretrained_encoder_path).encoder
        pretrain(dev, version, args.num_patients, encoder)
    else:
        exit("Mode not recognized!")
