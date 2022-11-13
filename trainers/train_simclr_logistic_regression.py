import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tqdm.notebook import tqdm

import constants
from datasets.SHHS_dataset_timeonly import SHHS_dataset_1, EEGdataModule

from models.logistic_regression import LogisticRegression
from models.simclr_model import CNNmodel_SimCLR
import torch
import torch.utils.data as data
from copy import deepcopy
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.load_and_save_model import load_model


def train_simclr_logistic_regression(args, device):
    dict_args = vars(args)
    model = load_model(CNNmodel_SimCLR, args.encoder_path)  # Load pretrained simclr model
    dm = EEGdataModule(**dict_args)  # Load datamodule
    dm.setup()

    train_ds = prepare_data_features(model, dm.train_dataloader(), device)
    val_ds = prepare_data_features(model, dm.val_dataloader(), device)
    test_ds = prepare_data_features(model, dm.test_dataloader(), device)

    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=args.num_workers)

    trainer = pl.Trainer(default_root_dir=os.path.join(args.CHECKPOINT_PATH, "LogisticRegression"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=args.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=False,
                         check_val_every_n_epoch=10)
    trainer.logger._default_hp_metric = None

    log_model = LogisticRegression(feature_dim=int(constants.SLEEP_EPOCH_SIZE/8*model.hparams.model_hparams['conv_filters'][-1]), **dict_args)
    trainer.fit(log_model, train_loader, val_loader)
    log_model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    train_result = trainer.test(log_model, train_loader, verbose=False)
    test_result = trainer.test(log_model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result


@torch.no_grad()
def prepare_data_features(model, data_loader, device):
    # Prepare model
    network = deepcopy(model.f)
    network.g = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    feats, labels = [], []
    for batch_inputs, batch_labels in tqdm(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_feats = network(batch_inputs.squeeze(dim=1))
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    return data.TensorDataset(feats, labels)