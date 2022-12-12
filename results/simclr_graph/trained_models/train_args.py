# This file contains the args used in the trained models


import constants


def get_data_args(first_patient_train=None, first_patient_test=None):
    return {
        "DATA_PATH": constants.SHHS_PATH_ESAT,
        "data_split": [4, 1],
        "first_patient_train": first_patient_train,
        "first_patient_test": first_patient_test,
        "num_patients_test": 50,
        "batch_size": 64,
        "num_workers": 12
    }


def get_logistic_args(data_path, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": "logistic_on_simclr",
        "DATA_PATH": data_path,
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


def get_supervised_args(data_path, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": "supervised_simclr",
        "DATA_PATH": data_path,
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


def get_finetune_args(data_path, checkpoint_path):
    return {
        "MODEL_TYPE": "SupervisedModel",
        "save_name": "finetuned_simclr",
        "DATA_PATH": data_path,
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
