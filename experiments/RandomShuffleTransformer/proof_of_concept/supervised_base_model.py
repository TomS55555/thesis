"""
    I will in this file first train a supervised base model where an encoder and outer transformer are trained
    in a fully supervised way on a small number of patients. This is the model with which the SSL method can be
    compared to.
"""

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
from models.cnn_transformer import CnnEncoder, FEAT_DIM
from models.sleep_transformer import OuterTransformer, Aggregator
from utils.helper_functions import load_model, get_data_path
from models.outer_supervised import OuterSupervisedModel
from models.simclr_transformer import SimCLR_Transformer
from models.random_shuffle_transformer import RandomShuffleTransformer

N_PATIENTS = 50

OUTER_DIM = 6  # Only 1 and 4 are supported at the moment

PATIENTS_PER_DS = 250  # Depends on RAM size of PC

train_path = "randomshuffle_trainings"  # path used for training the networks
result_file_name = "test_results_randomshuffle_transformer"


supervised_save_name = "fully_supervised_OT"

# parameters for SimCLR projection head
HIDDEN_DIM = 256
Z_DIM = 128


def get_CNN_encoder():
    """
        Input: batches of size [batch x outer x eeg-time]
        Output: batches of size [batch x outer x feat]
    """
    return nn.Sequential(
        CnnEncoder(input_size=constants.SLEEP_EPOCH_SIZE),
        # Aggregator(feat_dim=FEAT_DIM)
        nn.Flatten(),  # Output of size FEAT_DIM * 32 (last conv filter)
        nn.Linear(FEAT_DIM * 32, FEAT_DIM)
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

def get_classifier():
    return nn.Sequential(
        nn.Linear(FEAT_DIM, HIDDEN_DIM),
        nn.GELU(),
        nn.Linear(HIDDEN_DIM, constants.N_CLASSES)
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


def get_supervised_args(save_name, checkpoint_path, num_ds):
    return {
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": get_CNN_encoder(),
        "transformer": get_transformer(),
        "classifier": get_classifier(),

        "trainer_hparams": {
            "max_epochs": 30
            # "profiler": "simple"
        },
        "optim_hparams": {
            "lr": 1e-6,
            "weight_decay": 1e-9,
            "lr_hparams": None
        }
    }


def test_supervised(device, model):

    test_dm = EEGdataModule(test_set=True, **get_data_args(num_patients=5, batch_size=64, num_workers=0))

    trainer = pl.Trainer(
        default_root_dir=os.path.join(train_path, "testing"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        reload_dataloaders_every_n_epochs=0,
        # Reload dataloaders to get different part of the big dataset
        devices=1,  # How many GPUs/CPUs to use
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    temp_sup_res = list()
    for i in range(test_dm.n_test):
        test_dm.load_test_set(i)
        temp_sup_res.append(trainer.test(model=model,
                                         dataloaders=test_dm.test_dataloader()))
    sup_res = temp_sup_res
    return sup_res


def train_supervised(device, checkpoint_path, encoder, transformer, classifier, finetune_encoder, finetune_transformer):
    dm = EEGdataModule(**get_data_args(N_PATIENTS, batch_size=64))
    supervised_args = get_supervised_args(save_name=supervised_save_name,
                                          checkpoint_path=checkpoint_path,
                                          num_ds=dm.num_ds)
    supervised_model = OuterSupervisedModel(encoder=encoder,
                                       classifier=classifier,
                                        transformer=transformer,
                                       optim_hparams=supervised_args['optim_hparams'],
                                            finetune_encoder=finetune_encoder,
                                            finetune_transformer=finetune_transformer)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, supervised_save_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        reload_dataloaders_every_n_epochs=1 if dm.num_ds > 1 else 0,
        # Reload dataloaders to get different part of the big dataset
        devices=1,  # How many GPUs/CPUs to use
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
        **supervised_args['trainer_hparams']
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    trainer.fit(model=supervised_model,
                datamodule=dm)
    return supervised_model



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", required=False, default='0')
    parser.add_argument("--pretrained_encoder", required=False, default=None)
    parser.add_argument("--pretrained_transformer", required=False, default=None)
    parser.add_argument("--finetune_encoder", required=False, default=False)
    parser.add_argument("--finetune_transformer", required=False, default=False)
    parser.add_argument("--pretrained_classifier", required=False, default=None)
    args = parser.parse_args()

    encoder = load_model(SimCLR_Transformer, args.pretrained_encoder).f if args.pretrained_encoder is not None else get_CNN_encoder()
    transformer = load_model(RandomShuffleTransformer, args.pretrained_transformer).transformer if args.pretrained_transformer is not None else get_transformer()
    classifier = load_model(OuterSupervisedModel, args.pretrained_classifier).classifier if args.pretrained_classifier is not None else get_classifier()
    finetune_encoder = bool(args.finetune_encoder) if args.finetune_encoder is not None else False
    finetune_transformer = bool(args.finetune_transformer) if args.finetune_transformer is not None else False

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(dev)

    version = int(args.version)

    model = train_supervised(dev, train_path, encoder, transformer, classifier, finetune_encoder, finetune_transformer)
    result = test_supervised(dev, model)
    print(result)
