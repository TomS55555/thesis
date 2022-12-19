from datasets.datamodules import EEGdataModule, SimCLRdataModule
from models.supervised_model import SupervisedModel
from trainers.train_supervised import train_supervised
from argparse import Namespace
import constants
from utils.helper_functions import load_model, get_checkpoint_path, prepare_data_features
import pytorch_lightning as pl
import torch.utils.data as data
import os


def train_networks(pretrained_model, data_args, logistic_args, supervised_args, finetune_args, device):
    """
        This function can be used to train a sequence of a models: logistic, supervised and fine-tuned with a given pretrained encoder
    """
    dm = EEGdataModule(**data_args)  # Load datamodule

    # Run dm through pretrained encoder
    simclr_dm = SimCLRdataModule(pretrained_model, dm, data_args['batch_size'], data_args['num_workers'], device)

    # Train supervised model
    train_supervised(Namespace(**supervised_args), device, dm=dm)

    # Train logistic classifier on top of simclr backbone
    logistic_model = train_supervised(Namespace(**logistic_args), device=device, dm=simclr_dm)

    # Recover encoder from pretrained model for finetuning
    pretrained_encoder = type(pretrained_model.f)(**finetune_args['encoder_hparams'])
    pretrained_encoder.load_state_dict(pretrained_model.f.state_dict())

    # Use pretrained classifier as well for smooth learning: use the logistic result from above
    pretrained_classifier = type(logistic_model.classifier)(finetune_args['classifier_hparams']['input_dim'],
                                                            constants.N_CLASSES)
    pretrained_classifier.load_state_dict(logistic_model.classifier.state_dict())
    # Finally train the fine-tuned model
    train_supervised(Namespace(**finetune_args), device, dm=dm,
                     pretrained_encoder=pretrained_encoder,
                     pretrained_classifier=pretrained_classifier)


def test_networks(test_ds, pretrained_model, train_path, logistic_save_name, supervised_save_name, finetune_save_name, device, batch_size=64, num_workers=12):
    """
        Checkpoint path is the path for the testing
    """
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
        default_root_dir=os.path.join(train_path, "testing"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,  # How many GPUs/CPUs to use
        enable_progress_bar=True, )

    sup_model = load_model(SupervisedModel, get_checkpoint_path(train_path, supervised_save_name))
    sup_res = trainer.test(sup_model, dataloaders=test_dl)

    logistic_model = load_model(SupervisedModel, get_checkpoint_path(train_path, logistic_save_name))
    logistic_res = trainer.test(logistic_model, dataloaders=test_features_dl)

    fully_tuned_model = load_model(SupervisedModel, get_checkpoint_path(train_path, finetune_save_name))
    fully_tuned_res = trainer.test(fully_tuned_model, test_dl)

    return {
        "sup_res": sup_res,
        "logistic_res": logistic_res,
        "fully_tuned_res": fully_tuned_res
    }
