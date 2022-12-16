import sys
import os

sys.path.extend([os.getcwd()])
from datasets.datasets import EEGdataModule, SHHS_dataset_1
from models.simclr_model import SimCLR
import torch
from argparse import Namespace
from utils.helper_functions import load_model, SimCLRdataModule, prepare_data_features
from trainers.train_supervised import train_supervised
import constants
import json
from trained_models.train_args import get_data_args, get_logistic_args, get_finetune_args, get_supervised_args
import torch.utils.data as data
from models.supervised_model import SupervisedModel
import pytorch_lightning as pl
from argparse import ArgumentParser

encoder_path = "trained_models/cnn_simclr_500pat.ckpt"
pretrained_model = load_model(SimCLR, encoder_path)  # Load pretrained simclr model

patients_list = [3, 5, 10, 20, 50, 100, 250]  # n_patients used for training
first_patients_train = [1, 50, 100, 200]
first_patients_test = [1, 100, 200, 400]



def main_test(args):
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(device)
    n_patients_test = 50
    results = dict()
    for first_patient_test in first_patients_test:
        results_testset = dict()
        test_ds = SHHS_dataset_1(data_path=constants.SHHS_PATH_ESAT,
                                 first_patient=first_patient_test,
                                 num_patients=n_patients_test)

        for i in range(3):
            checkpoint_path = 'checkpoints_results' + str(i)
            results_testset['trained_models' + str(i)] = dict()
            for n_patients in patients_list:
                result = test_networks(test_dl=test_dl,
                                       test_features_dl=test_features_dl,
                                       n_patients_train=n_patients,
                                       checkpoint_path=checkpoint_path,
                                       device=device)
                results_testset['trained_models' + str(i)][str(n_patients)+"patients"] = result
        results["first_test_patient="+str(first_patient_test)] = results_testset

    with open("test_results.json", 'w+') as f:
        json.dump(results, f)









