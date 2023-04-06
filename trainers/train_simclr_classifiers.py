from datasets.datamodules import EEGdataModule, SimCLRdataModule
from models.supervised_model import SupervisedModel
from trainers.train_supervised import train_supervised
from argparse import Namespace
import constants
from utils.helper_functions import load_model, get_checkpoint_path, prepare_data_features
import pytorch_lightning as pl
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from copy import deepcopy
import os


def get_trainer(checkpoint_path, save_name, num_ds, trainer_hparams, device):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, save_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        reload_dataloaders_every_n_epochs=1 if num_ds > 1 else 0,
        # Reload dataloaders to get different part of the big dataset
        devices=1,  # How many GPUs/CPUs to use
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
        **trainer_hparams
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    return trainer


def train_networks(pretrained_model, data_args, logistic_args, supervised_args, finetune_args, device):
    """
        This function can be used to train a sequence of a models: logistic, supervised and fine-tuned with a given pretrained encoder
    """
    dm = EEGdataModule(**data_args)  # Load datamodule

    # Train supervised model
    # train_supervised(Namespace(**supervised_args), device, dm=dm)
    supervised_model = SupervisedModel(encoder=supervised_args['encoder'],
                                       classifier=supervised_args['classifier'],
                                       optim_hparams=supervised_args['optim_hparams'])

    supervised_trainer = get_trainer(checkpoint_path=supervised_args['CHECKPOINT_PATH'],
                                     save_name=supervised_args['save_name'],
                                     num_ds=dm.num_ds,
                                     trainer_hparams=supervised_args['trainer_hparams'],
                                     device=device)

    supervised_trainer.fit(model=supervised_model,
                           datamodule=dm)

    # Train logistic classifier on top of simclr backbone
    backbone = deepcopy(pretrained_model.f)
    for param in backbone.parameters():
        param.requires_grad = False

    logistic_model = SupervisedModel(encoder=backbone,
                                     classifier=logistic_args['classifier'],
                                     optim_hparams=logistic_args['optim_hparams'])

    logistic_trainer = get_trainer(checkpoint_path=logistic_args['CHECKPOINT_PATH'],
                                   save_name=logistic_args['save_name'],
                                   num_ds=dm.num_ds,
                                   trainer_hparams=logistic_args['trainer_hparams'],
                                   device=device)
    logistic_trainer.fit(model=logistic_model,
                         datamodule=dm)

    # Recover encoder from pretrained model for finetuning
    # pretrained_encoder = type(pretrained_model.f)(**finetune_args['encoder_hparams'])
    # pretrained_encoder.load_state_dict(pretrained_model.f.state_dict())

    pretrained_encoder = deepcopy(pretrained_model.f)
    pretrained_classifier = deepcopy(logistic_model.classifier)
    fine_tune_model = SupervisedModel(encoder=pretrained_encoder,
                                      classifier=pretrained_classifier,
                                      optim_hparams=finetune_args['optim_hparams'])
    fine_tune_trainer = get_trainer(checkpoint_path=finetune_args['CHECKPOINT_PATH'],
                                    save_name=finetune_args['save_name'],
                                    num_ds=dm.num_ds,
                                    trainer_hparams=finetune_args['trainer_hparams'],
                                    device=device)
    fine_tune_trainer.fit(model=fine_tune_model,
                          datamodule=dm)

    # Use pretrained classifier as well for smooth learning: use the logistic result from above
    # pretrained_classifier = type(logistic_model.classifier)(finetune_args['classifier_hparams']['input_dim'],
    #                                                        constants.N_CLASSES)
    # pretrained_classifier.load_state_dict(logistic_model.classifier.state_dict())
    # Finally train the fine-tuned model
    # train_supervised(Namespace(**finetune_args), device, dm=dm,
    #                  pretrained_encoder=pretrained_encoder,
    #                  pretrained_classifier=pretrained_classifier)


def test_networks(test_ds_args, train_path, logistic_save_name, supervised_save_name, finetune_save_name,
                  device):
    """
        Checkpoint path is the path for the testing
    """
    test_dm = EEGdataModule(test_set=True, **test_ds_args)
    trainer = get_trainer(
        checkpoint_path=train_path,
        save_name="testing",
        num_ds=test_dm.num_ds,
        trainer_hparams={},
        device=device
    )

    # print(list(iter(test_dm.test_dataloader()))[0][0].shape)

    sup_model = load_model(SupervisedModel, get_checkpoint_path(train_path, supervised_save_name))
    sup_res = trainer.test(model=sup_model,
                           datamodule=test_dm)

    logistic_model = load_model(SupervisedModel, get_checkpoint_path(train_path, logistic_save_name))
    logistic_res = trainer.test(model=logistic_model,
                                datamodule=test_dm)

    fully_tuned_model = load_model(SupervisedModel, get_checkpoint_path(train_path, finetune_save_name))
    fully_tuned_res = trainer.test(model=fully_tuned_model,
                                   datamodule=test_dm)

    return {
        "sup_res": sup_res,
        "logistic_res": logistic_res,
        "fully_tuned_res": fully_tuned_res
    }
