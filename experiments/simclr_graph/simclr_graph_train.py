from argparse import ArgumentParser

import torch
import constants

from models.simclr_model import SimCLR
from utils.helper_functions import load_model
from trainers.train_simclr_classifiers import train_networks, test_networks

encoder_path = "trained_models/cnn_simclr_500pat.ckpt"
pretrained_model = load_model(SimCLR, encoder_path)  # Load pretrained simclr model

patients_list = [3, 5, 10, 20, 50, 100, 250, 500]  # n_patients used for training

logistic_save_name = ""
supervised_save_name = ""
finetune_save_name = ""


def train(device):
    checkpoint_path = ''
    first_patients_train = 1

    for n_patients in patients_list:
        train_networks(
            pretrained_model=pretrained_model,
            data_args=get_data_args(first_patients_train, n_patients),
            logistic_args=get_logistic_args(logistic_save_name, checkpoint_path),
            supervised_args=get_supervised_args(supervised_save_name, checkpoint_path),
            finetune_args=get_finetune_args(finetune_save_name, checkpoint_path),
            device=device
        )


def test(device):
    checkpoint_path = ''
    first_patients_test = [1, 100, 200, 400]  # implicitly defines the test-set

    for first_patient_test in first_patients_test:
        test_networks()




def get_data_args(first_patient, num_patients):
    return {
        "data_path": constants.SHHS_PATH_ESAT,
        "data_split": [4, 1],
        "first_patient": first_patient,
        "num_patients": num_patients,
        "batch_size": 64,
        "num_workers": 12
    }


def get_logistic_args(save_name, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": "None",
        "encoder_hparams": {},

        "classifier": "logistic",
        "classifier_hparams": {
            "input_dim": 100
        },

        "trainer_hparams": {
            "max_epochs": 30
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


def get_supervised_args(save_name, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": "CNN_head",
        "encoder_hparams": {
            "conv_filters": [32, 64, 64],
            "representation_dim": 100
        },

        "classifier": "logistic",
        "classifier_hparams": {
            "input_dim": 100
        },

        "trainer_hparams": {
            "max_epochs": 40
        },
        "optim_hparams": {
            "lr": 1e-5,
            "weight_decay": 1e-3,
            "lr_hparams": None
        }
    }


def get_finetune_args(save_name, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": "CNN_head",
        "encoder_hparams": {
            "conv_filters": [32, 64, 64],
            "representation_dim": 100
        },

        "classifier": "logistic",
        "classifier_hparams": {
            "input_dim": 100
        },

        "trainer_hparams": {
            "max_epochs": 60
        },
        "optim_hparams": {
            "lr": 2e-6,
            "weight_decay": 0,
            "lr_hparams": None
        }
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(dev)

    if args.mode == "train":
        train(dev)
    elif args.mode == "test":
        test(dev)
    elif args.mode == 'both':
        train(dev)
        test(dev)
    else:
        exit("Mode not recognized!")
