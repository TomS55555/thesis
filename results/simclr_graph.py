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

result_file_name = "cnn_simclr_results.json"
checkpoint_path = 'checkpoints_results'
encoder_path = "trained_models/cnn_simclr01.ckpt"
pretrained_model = load_model(SimCLR, encoder_path)  # Load pretrained simclr model

patients_list = [3, 5, 10, 20, 50, 100] # n_patients used for training

data_args = {
  "DATA_PATH": "/esat/biomeddata/SHHS_Dataset/no_backup/",
  "data_split": [4, 1],
  "first_patient": 15,
  "num_patients_train": 15,
  "num_patients_test": 30,
  "batch_size": 64,
  "num_workers": 12
}

logistic_args = {
  "MODEL_TYPE": "SupervisedModel",
  "save_name": "logistic_on_simclr",
  "DATA_PATH": data_args['DATA_PATH'],
  "CHECKPOINT_PATH": checkpoint_path,

  "encoder": "None",
  "encoder_hparams": {},

  "classifier": "logistic",
  "classifier_hparams":{
      "input_dim": 100
  },
  "data_hparams": data_args,

  "trainer_hparams": {
    "max_epochs": 20
  },
  "optim_hparams": {
    "lr": 1e-3,
    "weight_decay": 1e-4
  }
}

supervised_args = {
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
    "max_epochs": 30
  },
  "optim_hparams": {
    "lr": 1e-4,
    "weight_decay": 1e-4
  }
}

finetune_logistic_args = {
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
  "classifier_hparams":{
      "input_dim": 100
  },
  "data_hparams": data_args,

  "trainer_hparams":{
    "max_epochs": 50
  },
  "optim_hparams": {
    "lr": 1e-6,
    "weight_decay": 0
  }
}


def train_networks(n_patients, device):
    data_args["num_patients_train"] = n_patients
    dm = EEGdataModule(**data_args)  # Load datamodule
    dm.setup()
    simclr_dm = SimCLRdataModule(pretrained_model, dm, data_args['batch_size'], data_args['num_workers'], device)

    supervised_args['save_name'] = "supervised_simclr" + '_'+str(n_patients) + '_patients'
    sup_model, sup_res = train_supervised(Namespace(**supervised_args), device, dm=dm)

    logistic_args['save_name'] = "logistic_on_simclr" + '_'+str(n_patients) + '_patients'
    logistic_model, logistic_res = train_supervised(Namespace(**logistic_args), device=device, dm=simclr_dm)

    finetune_logistic_args['save_name'] = "finetuned_simclr" + '_'+str(n_patients) + '_patients'

    pretrained_encoder = type(pretrained_model.f)(**finetune_logistic_args['encoder_hparams'])
    pretrained_encoder.load_state_dict(pretrained_model.f.state_dict())

    pretrained_classifier = type(logistic_model.classifier)(finetune_logistic_args['classifier_hparams']['input_dim'], constants.N_CLASSES)
    pretrained_classifier.load_state_dict(logistic_model.classifier.state_dict())

    fully_tuned_model, fully_tuned_res = train_supervised(Namespace(**finetune_logistic_args), device, dm=dm,
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
    results = dict()
    for n_patients in patients_list:
        result = train_networks(n_patients, device)
        results["n_patients"+"="+str(n_patients)] = result

    with open(result_file_name, 'w+') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()



