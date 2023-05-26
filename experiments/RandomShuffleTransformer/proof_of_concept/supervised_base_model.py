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
import json
from copy import deepcopy


OUTER_DIM = 6

PATIENTS_PER_DS = 250  # Depends on RAM size of PC

# parameters for projection head
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


def get_supervised_args(save_name, checkpoint_path):
    return {
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": None,
        "transformer": None,
        "classifier": None,

        "trainer_hparams": {
            "max_epochs": 30
            # "profiler": "simple"
        },
        "optim_hparams": {
            "lr": 1e-5,
            "weight_decay": 1e-9,
            "lr_hparams": None
        }
    }

def get_logistic_args(save_name, checkpoint_path):
    return {
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": None,

        "classifier": get_classifier(),

        "trainer_hparams": {
            "max_epochs": 80,
        },
        "optim_hparams": {
            "lr": 1e-4,
            "weight_decay": 0,
            "lr_hparams": None
        }
    }

def get_finetune_args(save_name, checkpoint_path):
    return {
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": None,
        "classifier": None,

        "trainer_hparams": {
            "max_epochs": 40
        },
        "optim_hparams": {
            "lr": 5e-6,
            "weight_decay": 0,
            "lr_hparams": None
        }
    }

def test_supervised(device, model, checkpoint_path: str, test_path):

    test_dm = EEGdataModule(test_set=True, **get_data_args(num_patients=5, batch_size=64, num_workers=0))

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, test_path),
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


def train_supervised(device, num_patients: int,
                     encoder: nn.Module, transformer: nn.Module,
                     classifier: nn.Module, finetune_encoder: bool, finetune_transformer: bool, args):

    dm = EEGdataModule(**get_data_args(num_patients, batch_size=64))
    supervised_model = OuterSupervisedModel(encoder=encoder,
                                       classifier=classifier,
                                        transformer=transformer,
                                       optim_hparams=args['optim_hparams'],
                                            finetune_encoder=finetune_encoder,
                                            finetune_transformer=finetune_transformer)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(args['CHECKPOINT_PATH'], args['save_name']),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        reload_dataloaders_every_n_epochs=1 if dm.num_ds > 1 else 0,
        # Reload dataloaders to get different part of the big dataset
        devices=1,  # How many GPUs/CPUs to use
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
        **args['trainer_hparams']
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    trainer.fit(model=supervised_model,
                datamodule=dm)
    return supervised_model



def train_models_n_pat(device, num_patients: int, save_name: str, checkpoint_path: str, pretrained_encoder: nn.Module, pretrained_transformer: nn.Module, result_file_name=None):
    """
        This function uses a pretrained encoder and pretrained transformer to conduct the following experiment
        1) Fix both pretrained encoder and pretrained outer transformer and train a classifier on top
        2) Fine-tune the whole thing
        3) Compare it with a fully supervised method with the same amount of patients
    """
    save_name_logistic = save_name + 'logistic' + str(num_patients)+'pat'
    save_name_finetune = save_name + 'fine_tuned' + str(num_patients) + 'pat'
    save_name_supervised = save_name + 'supervised' + str(num_patients) + 'pat'

    logistic_model = train_supervised(device=device,
                                      num_patients=num_patients,
                                      encoder=deepcopy(pretrained_encoder),
                                      transformer=deepcopy(pretrained_transformer),
                                      classifier=get_classifier(),
                                      finetune_encoder=False,
                                      finetune_transformer=False,
                                      args=get_logistic_args(save_name_logistic, checkpoint_path))

    finetune_model = train_supervised(device=device,
                                      num_patients=num_patients,
                                      encoder=deepcopy(pretrained_encoder),
                                      transformer=deepcopy(pretrained_transformer),
                                      classifier=get_classifier(),
                                      finetune_encoder=True,
                                      finetune_transformer=True,
                                      args=get_finetune_args(save_name_finetune, checkpoint_path))

    fully_supervised_model = train_supervised(device=device,
                                              num_patients=num_patients,
                                              encoder=get_CNN_encoder(),
                                              transformer=get_transformer(),
                                              classifier=get_classifier(),
                                              finetune_encoder=True,
                                              finetune_transformer=True,
                                              args=get_supervised_args(save_name_supervised, checkpoint_path))

    test_res_logistic = test_supervised(device, logistic_model, checkpoint_path, save_name_logistic)
    test_res_finetuned = test_supervised(device, finetune_model, checkpoint_path, save_name_finetune)
    test_res_supervised = test_supervised(device, fully_supervised_model, checkpoint_path, save_name_supervised)

    results = {
        "sup_res": test_res_supervised,
        "logistic_res": test_res_logistic,
        "fully_tuned_res": test_res_finetuned
    }
    print(results)
    if save_name is not None:
        with open(save_name, 'w+') as fp:
            json.dump(results, fp)
    return results


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
    train_models_n_pat(dev,
                       num_patients=50,
                       save_name='test',
                       checkpoint_path='test',
                       pretrained_encoder=get_CNN_encoder(),
                       pretrained_transformer=get_transformer(),
                       result_file_name='result_test')
    #model = train_supervised(dev, train_path, encoder, transformer, classifier, finetune_encoder, finetune_transformer)
    #result = test_supervised(dev, model)
    #print(result)
