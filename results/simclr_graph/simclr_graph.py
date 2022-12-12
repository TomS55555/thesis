import sys
import os

sys.path.extend([os.getcwd()])
from datasets.SHHS_dataset_timeonly import EEGdataModule, SHHS_dataset_1
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
from pytorch_lightning as pl
from argparse import ArgumentParser

encoder_path = "trained_models/cnn_simclr_500pat.ckpt"
pretrained_model = load_model(SimCLR, encoder_path)  # Load pretrained simclr model

patients_list = [3, 5, 10, 20, 50, 100, 250]  # n_patients used for training
first_patients_train = [1, 50, 100, 200]
first_patients_test = [1, 100, 200, 400]

def train_networks(checkpoint_path, first_patient_train, n_patients, device):
    data_args = get_data_args(first_patient_train=first_patient_train)
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


def test_networks(first_patient_test, n_patients, checkpoint_path, data_path, device, batch_size=64, num_workers=12):
    """
        1) Define test set
        2) Load models
        3) Run test set on models
        4) Return result in nice format
    """

    test_ds = SHHS_dataset_1(data_path=data_path,
                             first_patient=first_patient_test,
                             num_patients=n_patients)

    test_dl = data.DataLoader(dataset=test_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    test_features_ds = prepare_data_features(model=pretrained_model,
                                             data_loader=test_dl,
                                             device=device)

    test_features_dl = data.DataLoader(dataset=test_features_ds,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, 'testing'),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,  # How many GPUs/CPUs to use
        enable_progress_bar=True,)

    save_name_sup = "supervised_simclr" + '_' + str(n_patients) + '_patients'
    sup_model = load_model(SupervisedModel, get_checkpoint_path(checkpoint_path, save_name_sup))
    sup_res = trainer.test(sup_model, dataloaders=test_dl)

    save_name_logistic = "logistic_on_simclr" + '_' + str(n_patients) + '_patients'
    logistic_model = load_model(SupervisedModel, get_checkpoint_path(checkpoint_path, save_name_logistic))
    logistic_res = trainer.test(logistic_model, dataloaders=test_features_dl)

    save_name_finetuned = "finetuned_simclr" + '_' + str(n_patients) + '_patients'
    fully_tuned_model = load_model(SupervisedModel, get_checkpoint_path(checkpoint_path, save_name_finetuned))

    fully_tuned_res = trainer.test(fully_tuned_model, test_dl)


    return {
        "sup_res": sup_res,
        "logistic_res": logistic_res,
        "fully_tuned_res": fully_tuned_res
    }

def get_checkpoint_path(checkpoint_path, save_name):
    rest_path = 'lightning_logs/version_0/checkpoints'
    dir_path = os.path.join(checkpoint_path, save_name, rest_path)
    dirs = os.listdir(dir_path)
    ckpt = list(filter(lambda x: x.startswith("epoch"), dirs))[0]
    print("Found checkpoint: ", ckpt)
    return os.path.join(dir_path, ckpt)

def main(args):
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print(device)
    for i in range(4):
        result_file_name = "cnn_simclr_results"+str(i)+"_"+args.mode+".json"
        checkpoint_path = 'checkpoints_results'+str(i)

        results = dict()
        for n_patients in patients_list:
            print("test ", n_patients)
            result = ""
            if args.mode == "train":
                result = train_networks(checkpoint_path=checkpoint_path,
                                        first_patient_train=first_patients_train[i],
                                        n_patients=n_patients,
                                        device=device)
            elif args.mode == "test":
                result = test_networks(first_patient_test=first_patients_test[i],
                                       n_patients=50,
                                       checkpoint_path=checkpoint_path,
                                       data_path=constants.SHHS_PATH_ESAT,
                                       device=device)
            results["n_patients" + "=" + str(n_patients)+"_"+args.mode] = result

        with open(result_file_name, 'w+') as f:
            json.dump(results, f)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", required=True)
    main(parser.parse_args())
