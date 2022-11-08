import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.SHHS_dataset_timeonly import EEGdataModule
from datasets.augmentations import ContrastiveTransformations, AmplitudeScale, TimeShift, ZeroMask, GaussianNoise
from models.conv_model import CNNmodel_SimCLR, CNNmodel_supervised


def train_cnn_supervised(args, device):
    save_name = args.model_name
    dict_args = vars(args)
    pl.seed_everything(42)  # To be reproducable

    data_module = EEGdataModule(**dict_args)
    data_module.setup()

    trainer = Trainer.from_argparse_args(args,
                                         default_root_dir = os.path.join(args.CHECKPOINT_PATH, save_name),
                                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                                         devices=1,  # How many GPUs/CPUs to use
                                         callbacks=[
                                             ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc", save_last=True),
                                             # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                             LearningRateMonitor("epoch")],  # Log learning rate every epoch
                                         enable_progress_bar=True
                                         )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(args.CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = CNNmodel_supervised.load_from_checkpoint(
            pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        model = CNNmodel_supervised(**dict_args)
        trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
        model = CNNmodel_supervised.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training


    # Test best model on validation and test set
    val_result = trainer.test(model, data_module.val_dataloader(), verbose=False)
    test_result = trainer.test(model, data_module.test_dataloader(), verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


def train_simclr(args, device):
    save_name = args.model_name
    dict_args = vars(args)
    pl.seed_everything(42)  # To be reproducable

    p = 0.2  # probability of applying the transforms
    contrast_transforms = ContrastiveTransformations(
        [
            AmplitudeScale(dict_args["amplitude-min"], dict_args["amplitude-max"], p, 1),
            GaussianNoise(dict_args["noise-min"], dict_args["noise-max"], p, 1),
            ZeroMask(dict_args["zeromask-min"], dict_args["zeromask-max"], p, 1),
            TimeShift(dict_args["timeshift-min"], dict_args["timeshift-max"], p, 1)
        ], n_views=2
    )
    data_module = EEGdataModule(transform=contrast_transforms, **dict_args)
    data_module.setup()
    trainer = Trainer.from_argparse_args(args,
                                         default_root_dir=os.path.join(args.CHECKPOINT_PATH, save_name),
                                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                                         devices=1,  # How many GPUs/CPUs to use
                                         callbacks=[
                                             ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5",
                                                             save_last=True),
                                             # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                             LearningRateMonitor("epoch")],  # Log learning rate every epoch
                                         enable_progress_bar=True
                                         )

    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(args.CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = CNNmodel_SimCLR.load_from_checkpoint(
            pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        model = CNNmodel_SimCLR(**dict_args)
        trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
        model = CNNmodel_SimCLR.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, data_module.val_dataloader(), verbose=False)
    test_result = trainer.test(model, data_module.test_dataloader(), verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    return model, result
