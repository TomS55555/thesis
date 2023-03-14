import sys, os
sys.path.extend([os.getcwd()])
from argparse import ArgumentParser
import torch
import constants
import json

from models.simclr_model import SimCLR
from utils.helper_functions import load_model
from trainers.train_simclr_classifiers import train_networks, test_networks
from datasets.datasets import SHHSdataset
from models.mymodules import CNN_head
import torch.nn as nn
import math

encoder_path = "trained_models/cnn_model_5000pat/last.ckpt"
pretrained_model = load_model(SimCLR, encoder_path)  # Load pretrained simclr model

#patients_list = [3, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000]  # n_patients used for training
patients_list = [3, 5, 10, 20, 50, 100]

# patients_list = [50]

train_path = "simclr_trainings"  # path used for training the networks
result_file_name = "test_results"

logistic_save_name = "logistic_on_simclr"
supervised_save_name = "fully_supervised"
finetune_save_name = "fine_tuned_simclr"

PATIENTS_PER_DS = 250  # seems to work for google cloud


def train(device, version):
    first_patients_train = 1

    for n_patients in patients_list:
        num_ds = math.ceil(n_patients/PATIENTS_PER_DS)  
        train_networks(
            pretrained_model=pretrained_model,
            data_args=get_data_args(first_patients_train, n_patients, num_ds),
            logistic_args=get_logistic_args(logistic_save_name+"_"+str(n_patients)+"pat", train_path+str(version), num_ds),
            supervised_args=get_supervised_args(supervised_save_name+"_"+str(n_patients)+"pat", train_path+str(version), num_ds),
            finetune_args=get_finetune_args(finetune_save_name+"_"+str(n_patients)+"pat", train_path+str(version), num_ds),
            device=device
        )


def test(device, version):
    results = dict()

    for n_patients in patients_list:
        test_results = test_networks(
            encoder=CNN_head(
                conv_filters=[32, 64, 64],
                representation_dim=100
            ),
            classifier=nn.Linear(
                in_features=100,
                out_features=constants.N_CLASSES
            ),
            test_ds_args=get_data_args(first_patient=1, num_patients=5),  #TODO: LOOK INTO THIS!!
            train_path=train_path+str(version),
            logistic_save_name=logistic_save_name+"_"+str(n_patients)+"pat",
            supervised_save_name=supervised_save_name+"_"+str(n_patients)+"pat",
            finetune_save_name=finetune_save_name+"_"+str(n_patients)+"pat",
            device=device
        )
        results[str(n_patients)+"_pat"] = test_results
        print(test_results)
    print(results)
    with open(result_file_name+str(version), 'w+') as f:
        json.dump(results, f)


def get_data_args(first_patient, num_patients, num_ds):
    return {
        "data_path": constants.SHHS_PATH_GOOGLE,
        "data_split": [4, 1],
        "first_patient": first_patient,
        "num_patients": num_patients,
        "batch_size": 64,
        "num_workers": 2,
        "num_ds": num_ds,
        "exclude_test_set": constants.TEST_SET_1
    }


def get_logistic_args(save_name, checkpoint_path, num_ds):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": None,

        "classifier": nn.Linear(
            in_features=100,
            out_features=constants.N_CLASSES
        ),

        "trainer_hparams": {
            "max_epochs": 30 * num_ds,
            #'profiler': 'simple'
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
        "MODEL_TYPE": "SupervisedModel",
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": CNN_head(
            conv_filters=[32, 64, 64],
            representation_dim=100
        ),
        "classifier": nn.Linear(
            in_features=100,
            out_features=constants.N_CLASSES
        ),

        "trainer_hparams": {
            "max_epochs": 40 * num_ds
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
        "MODEL_TYPE": "SupervisedModel",
        "save_name": save_name,
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": CNN_head(
            conv_filters=[32, 64, 64],
            representation_dim=100
        ),
        "classifier": nn.Linear(
            in_features=100,
            out_features=constants.N_CLASSES
        ),

        "trainer_hparams": {
            "max_epochs": 60 * num_ds
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
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(dev)

    version = int(args.version)

    if args.mode == "train":
        train(dev, version)
    elif args.mode == "test":
        test(dev, version)
    elif args.mode == 'both':
        train(dev, version)
        test(dev, version)
    else:
        exit("Mode not recognized!")
