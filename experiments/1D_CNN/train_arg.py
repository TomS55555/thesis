import argparse
import sys
import os
sys.path.extend([os.getcwd()])
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from models.conv_model import CNNmodel
from pytorch_lightning import Trainer
from datasets.SHHS_dataset_timeonly import EEGdataModule
import json


def train(args):
    save_name = args.model_name
    dict_args = vars(args)
    pl.seed_everything(42) # To be reproducable

    data_module = EEGdataModule(**dict_args)
    data_module.setup()

    trainer = Trainer.from_argparse_args(args,
                                         default_root_dir = os.path.join(args.CHECKPOINT_PATH, save_name),
                                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                                         devices=1,  # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                                         callbacks=[
                                             ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc", save_last=True),
                                             # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                             LearningRateMonitor("epoch")],  # Log learning rate every epoch
                                         enable_progress_bar=True  # Set to False if you do not want a progress bar
                                         )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(args.CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = CNNmodel.load_from_checkpoint(
            pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        model = CNNmodel(**dict_args)
        trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
        model = CNNmodel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training


    # Test best model on validation and test set
    val_result = trainer.test(model, data_module.val_dataloader(), verbose=False)
    test_result = trainer.test(model, data_module.test_dataloader(), verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)
    parser = ArgumentParser()
    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    # add PROGRAM level args
    parser.add_argument("--model_name", type=str, default="cnn_model01")

    # add model specific args
    parser = CNNmodel.add_model_specific_args(parser)

    # add data specific args
    parser = EEGdataModule.add_argparse_args(parser)

    parser.add_argument("--load_json", help="Load settings from file json format. Command line options override values in file.")

    args = parser.parse_args()

    if args.load_json:
        with open(args.load_json, 'rt') as f:
            args = parser.parse_args(namespace=argparse.Namespace(**json.load(f)))

    mod, res = train(args)
    print(res)
