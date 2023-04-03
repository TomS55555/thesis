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
from datasets.datasets import SHHS_dataset_STFT
import pytorch_lightning as pl
from models.sleep_transformer import InnerTransformer
from models.supervised_model import SupervisedModel
from models.simclr_model import SimCLR
from utils.helper_functions import load_model, get_data_path
from datasets.augmentations import AugmentationModuleSTFT
from trainers.train_simclr_classifiers import train_networks, test_networks
import json

patients_list = [5, 10, 20, 50, 100]

OUTER_DIM_STFT = 1  # Only pretraining inner transformer

PATIENTS_PER_DS = 250  # Depends on RAM size of PC

train_path = "simclr_trainings"  # path used for training the networks
result_file_name = "test_results"

pretrained_save_name = "pretrained_IT"
logistic_save_name = "logistic_on_simclr_IT"
supervised_save_name = "fully_supervised_IT"
finetune_save_name = "fine_tuned_simclr_IT"

# parameters for SimCLR projection head
HIDDEN_DIM = 256
Z_DIM = 128

MAX_EPOCHS = 100  # max epochs independent of number of datasets


def get_encoder():
    return nn.Sequential(
        InnerTransformer(
            inner_dim=constants.INNER_DIM_STFT,
            feat_dim=constants.FEAT_DIM_STFT,
            dim_feedforward=1024,
            num_heads=8,
            num_layers=4
        ),
        nn.Flatten()  # this should result in a final layer of size outer x feat = feat bc outer=1 for pretraining
    )


def get_projection_head():
    return nn.Sequential(
        nn.Linear(constants.FEAT_DIM_STFT, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, Z_DIM)
    )


def get_classifier():
    return nn.Linear(
        constants.FEAT_DIM_STFT,
        constants.N_CLASSES
    )


def get_data_args(num_patients, batch_size):
    return {
        "data_path": get_data_path(),
        "data_split": [4, 1],
        "first_patient": 1,
        "num_patients": num_patients,
        "batch_size": batch_size,
        "num_workers": 4,
        "num_ds": math.ceil(num_patients / PATIENTS_PER_DS),
        "exclude_test_set": constants.TEST_SET_1,
        "dataset_type": SHHS_dataset_STFT,
        "window_size": OUTER_DIM_STFT
    }


def get_logistic_args(save_name, checkpoint_path, num_ds):
    return {
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": None,

        "classifier": get_classifier(),

        "trainer_hparams": {
            "max_epochs": min(MAX_EPOCHS, 30 * num_ds),
        },
        "optim_hparams": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lr_hparams": {
                "gamma": 0.1,
                "milestones": [10]
            }
        }
    }


def get_supervised_args(save_name, checkpoint_path, num_ds):
    return {
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": get_encoder(),
        "classifier": get_classifier(),

        "trainer_hparams": {
            "max_epochs": min(40 * num_ds, MAX_EPOCHS)
            # "profiler": "simple"
        },
        "optim_hparams": {
            "lr": 1e-5,
            "weight_decay": 1e-3,
            "lr_hparams": None
        }
    }


def get_finetune_args(save_name, checkpoint_path, num_ds):
    return {
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": get_encoder(),
        "classifier": get_classifier(),

        "trainer_hparams": {
            "max_epochs": min(60 * num_ds, MAX_EPOCHS)
        },
        "optim_hparams": {
            "lr": 2e-6,
            "weight_decay": 0,
            "lr_hparams": None
        }
        }


def pretrain(device, version):
    #TODO: fix normalization of STFT images!
    num_patients = 5000
    batch_size = 512
    max_epochs = 100
    dm = EEGdataModule(**get_data_args(num_patients=num_patients,
                                       batch_size=batch_size))
    model = SimCLR(
        aug_module=AugmentationModuleSTFT(
            batch_size=batch_size,
            time_mask_window=15,
            freq_mask_window=60
        ),
        encoder=get_encoder(),
        projector=get_projection_head(),
        temperature=0.05,
        optim_hparams={
            "max_epochs": max_epochs,
            "lr": 3e-4,
            "weight_decay": 1e-4
        }
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


def train(pretrained_model, device, version):
    for n_patients in patients_list:
        num_ds = math.ceil(n_patients / PATIENTS_PER_DS)
        train_networks(
            pretrained_model=pretrained_model,
            data_args=get_data_args(n_patients, num_ds),
            logistic_args=get_logistic_args(logistic_save_name + "_" + str(n_patients) + "pat",
                                            train_path + str(version), num_ds),
            supervised_args=get_supervised_args(supervised_save_name + "_" + str(n_patients) + "pat",
                                                train_path + str(version), num_ds),
            finetune_args=get_finetune_args(finetune_save_name + "_" + str(n_patients) + "pat",
                                            train_path + str(version), num_ds),
            device=device
        )

    # Then train logistic classifier on top of pretrained transformer


def test(device, version):
    results = dict()

    for n_patients in patients_list:
        test_results = test_networks(
            encoder=get_encoder(),
            classifier=get_classifier(),
            test_ds_args=get_data_args(num_patients=5,
                                       batch_size=64),  # TODO: LOOK INTO THIS!!
            train_path=train_path + str(version),
            logistic_save_name=logistic_save_name + "_" + str(n_patients) + "pat",
            supervised_save_name=supervised_save_name + "_" + str(n_patients) + "pat",
            finetune_save_name=finetune_save_name + "_" + str(n_patients) + "pat",
            device=device
        )
        results[str(n_patients) + "_pat"] = test_results
        print(test_results)
    print(results)
    with open(result_file_name + str(version), 'w+') as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--version", required=False, default="0")
    parser.add_argument("--pretrained_path", required=False, default=None)
    args = parser.parse_args()

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(dev)

    version = int(args.version)

    if args.mode == "pretrain":
        pretrain(dev, version)
    elif args.mode == "train":
        if args.pretrained_path is None:
            print("A pretrained encoder is required, specify it with the --pretrained_path")
            sys.exit(1)
        pretrained_model = load_model(SimCLR, args.pretrained_path)
        train('', dev, version)
    elif args.mode == "test":
        test(dev, version)
    elif args.mode == 'both':
        pretrained_model = load_model(SimCLR, args.pretrained_path) if args.pretrained_path is not None else pretrain(dev, version)
        train('', dev, version)
        test(dev, version)
    else:
        exit("Mode not recognized!")
