import sys
import os

sys.path.extend([os.getcwd()])
from datasets.SHHS_dataset_timeonly import EEGdataModule
from models.simclr_model import SimCLR
import torch
from argparse import Namespace
from utils.helper_functions import load_model, SimCLRdataModule
from trainers.train_supervised import train_supervised
import constants
import json

encoder_path = "trained_models/cnn_simclr01.ckpt"
pretrained_model = load_model(SimCLR, encoder_path)  # Load pretrained simclr model

patients_list = [3, 5, 10, 20, 50, 100, 250]  # n_patients used for training


def get_data_args(first_patient):
    return {
        "DATA_PATH": "/esat/biomeddata/SHHS_Dataset/no_backup/",
        "data_split": [4, 1],
        "first_patient": first_patient,
        "num_patients_train": 15,
        "num_patients_test": 30,
        "batch_size": 64,
        "num_workers": 12
    }


def get_logistic_args(data_args, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": "logistic_on_simclr",
        "DATA_PATH": data_args['DATA_PATH'],
        "CHECKPOINT_PATH": checkpoint_path,

        "encoder": "None",
        "encoder_hparams": {},

        "classifier": "logistic",
        "classifier_hparams": {
            "input_dim": 100
        },
        "data_hparams": data_args,

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


def get_supervised_args(data_args, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": "supervised_simclr",
        "DATA_PATH": data_args['DATA_PATH'],
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
        "data_hparams": data_args,

        "trainer_hparams": {
            "max_epochs": 40
        },
        "optim_hparams": {
            "lr": 1e-5,
            "weight_decay": 1e-3,
            "lr_hparams": None
        }
    }


def get_finetune_args(data_args, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": "finetuned_simclr",
        "DATA_PATH": data_args['DATA_PATH'],
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
        "data_hparams": data_args,

        "trainer_hparams": {
            "max_epochs": 60
        },
        "optim_hparams": {
            "lr": 2e-6,
            "weight_decay": 0,
            "lr_hparams": None
        }
    }


def train_networks(checkpoint_path, first_patient, n_patients, device):
    data_args = get_data_args(first_patient)
    data_args["num_patients_train"] = n_patients

    logistic_args = get_logistic_args(data_args, checkpoint_path)
    finetune_args = get_finetune_args(data_args, checkpoint_path)
    supervised_args = get_supervised_args(data_args, checkpoint_path)

    dm = EEGdataModule(**data_args)  # Load datamodule
    dm.setup()
    simclr_dm = SimCLRdataModule(pretrained_model, dm, data_args['batch_size'], data_args['num_workers'], device)

    supervised_args['save_name'] = "supervised_simclr" + '_' + str(n_patients) + '_patients'
    sup_model, sup_res = train_supervised(Namespace(**supervised_args), device, dm=dm)

    logistic_args['save_name'] = "logistic_on_simclr" + '_' + str(n_patients) + '_patients'
    logistic_model, logistic_res = train_supervised(Namespace(**logistic_args), device=device, dm=simclr_dm)

    finetune_args['save_name'] = "finetuned_simclr" + '_' + str(n_patients) + '_patients'

    pretrained_encoder = type(pretrained_model.f)(**finetune_args['encoder_hparams'])
    pretrained_encoder.load_state_dict(pretrained_model.f.state_dict())

    pretrained_classifier = type(logistic_model.classifier)(finetune_args['classifier_hparams']['input_dim'],
                                                            constants.N_CLASSES)
    pretrained_classifier.load_state_dict(logistic_model.classifier.state_dict())

    fully_tuned_model, fully_tuned_res = train_supervised(Namespace(**finetune_args), device, dm=dm,
                                                          pretrained_encoder=pretrained_encoder,
                                                          pretrained_classifier=pretrained_classifier)
    return {
        "sup_res": sup_res,
        "logistic_res": logistic_res,
        "fully_tuned_res": fully_tuned_res
    }


def main():
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(device)
    for i in range(3):
        result_file_name = "cnn_simclr_results"+str(i)+".json"
        checkpoint_path = 'checkpoints_results'+str(i)
        first_patient = 50 * i + 7  # This implicitly defines the test-set, play with it and average results

        results = dict()
        for n_patients in patients_list:
            result = train_networks(checkpoint_path=checkpoint_path,
                                    first_patient=first_patient,
                                    n_patients=n_patients,
                                    device=device)
            results["n_patients" + "=" + str(n_patients)] = result

        with open(result_file_name, 'w+') as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
