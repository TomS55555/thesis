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
from models.feat_predictor import FeatPredictor

patients_list = [3, 5, 10, 20, 50, 100, 250]

OUTER_DIM = 4  # Only 1 and 4 are supported at the moment

PATIENTS_PER_DS = 250  # Depends on RAM size of PC

train_path = "featpredictor_trainings"  # path used for training the networks
result_file_name = "test_results_cnn_transformer"

pretrained_save_name = "pretrained_IT"
logistic_save_name = "logistic_on_simclr_IT"
supervised_save_name = "fully_supervised_IT"
finetune_save_name = "fine_tuned_simclr_IT"

# parameters for SimCLR projection head
HIDDEN_DIM = 256
Z_DIM = 128

MAX_EPOCHS = 100  # max epochs independent of number of datasets


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
        The input to the projection head is the output of the aggregator:
        [batch x feat] and the output is also of size [batch x feat]
    """
    return nn.Sequential(nn.Linear(FEAT_DIM, HIDDEN_DIM),
                         nn.GELU(),
                         nn.Linear(HIDDEN_DIM, FEAT_DIM))


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
        "exclude_test_set": constants.TEST_SET_1,
        "dataset_type": SHHSdataset,
        "window_size": OUTER_DIM
    }


def get_logistic_args(save_name, checkpoint_path, num_ds):
    return {
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": None,

        "classifier": get_classifier(),

        "trainer_hparams": {
            "max_epochs": min(MAX_EPOCHS, 40 * num_ds),
        },
        "optim_hparams": {
            "lr": 1e-4,
            "weight_decay": 0,
            "lr_hparams": None
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
            "lr": 1e-4,
            "weight_decay": 0,
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
            "max_epochs": min(40 * num_ds, MAX_EPOCHS)  # TODO: change back to 60
        },
        "optim_hparams": {
            "lr": 1e-5,
            "weight_decay": 0,
            "lr_hparams": None
        }
    }


def pretrain(device, version, encoder=None):
    # TODO: fix normalization of STFT images!
    num_patients = 20
    batch_size = 64
    max_epochs = 10
    dm = EEGdataModule(**get_data_args(num_patients=num_patients,
                                       batch_size=batch_size))
    model = FeatPredictor(
        encoder=get_encoder() if encoder is None else encoder,
        transformer=get_transformer(),
        proj_head=get_projection_head(),
        optim_hparams={
            "max_epochs": max_epochs,
            "lr": 5e-5,
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


def train(pretrained_model, device, version, train_supervised=False):
    for n_patients in patients_list:
        num_ds = math.ceil(n_patients / PATIENTS_PER_DS)
        train_networks(
            pretrained_model=pretrained_model,
            data_args=get_data_args(n_patients, batch_size=64),
            logistic_args=get_logistic_args(logistic_save_name + "_" + str(n_patients) + "pat",
                                            train_path + str(version), num_ds),
            supervised_args=get_supervised_args(supervised_save_name + "_" + str(n_patients) + "pat",
                                                train_path + str(version), num_ds),
            finetune_args=get_finetune_args(finetune_save_name + "_" + str(n_patients) + "pat",
                                            train_path + str(version), num_ds),
            train_supervised=train_supervised,
            device=device
        )

    # Then train logistic classifier on top of pretrained transformer


def test(pretrained_model, device, version, test_supervised=False):
    results = dict()

    for n_patients in patients_list:
        test_results = test_networks(
            pretrained_model=pretrained_model,
            test_ds_args=get_data_args(num_patients=5,
                                       batch_size=64,
                                       num_workers=0),  # TODO: LOOK INTO THIS!!
            train_path=train_path + str(version),
            logistic_save_name=logistic_save_name + "_" + str(n_patients) + "pat",
            supervised_save_name=supervised_save_name + "_" + str(n_patients) + "pat",
            finetune_save_name=finetune_save_name + "_" + str(n_patients) + "pat",
            device=device,
            test_supervised=test_supervised
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
    parser.add_argument("--pretrained_encoder_path", required=False, default=None)
    args = parser.parse_args()

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(dev)

    version = int(args.version)

    ENCODER_TYPE = nn.Sequential(
        CnnEncoder(input_size=constants.SLEEP_EPOCH_SIZE),
        #Aggregator(feat_dim=FEAT_DIM)
        nn.Flatten(), # Output of size FEAT_DIM * 32 (last conv filter)
        nn.Linear(FEAT_DIM*32, FEAT_DIM)
    )

    if args.mode == "pretrain":
        encoder = load_model(SimCLR_Transformer, args.pretrained_encoder_path).f
        pretrain(dev, version, encoder)
    elif args.mode == "train":
        if args.pretrained_path is None:
            print("A pretrained encoder is required, specify it with the --pretrained_path")
            sys.exit(1)
        pretrained_model = load_model(SimCLR_Transformer, args.pretrained_path)
        train(pretrained_model, dev, version, True)
    elif args.mode == "test":
        pretrained_model = load_model(SimCLR_Transformer, args.pretrained_path)
        test(pretrained_model, dev, version, True)
    elif args.mode == 'both':
        pretrained_model = load_model(SimCLR_Transformer, args.pretrained_path) if args.pretrained_path is not None else pretrain(
            dev, version)
        train(pretrained_model, dev, version, True)
        test(pretrained_model, dev, version, True)
    else:
        exit("Mode not recognized!")
