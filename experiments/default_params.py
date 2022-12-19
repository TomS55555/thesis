"""
    Some default args are saved here for easy testing
"""

import constants


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