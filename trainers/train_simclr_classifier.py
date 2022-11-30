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
from utils.helper_functions import load_model


def train_simclr_classifier(args, device):
    dict_args = vars(args)
    pretrained_filename = os.path.join(dict_args['CHECKPOINT_PATH'], "simclr01.ckpt") # load trained simclr model

    model = load_model(CNNmodel_SimCLR, pretrained_filename)


    results = {}
    for num_patients_per_label in [1, 5, 10, 100]:
        dm = EEGdataModule(DATA_PATH=args.DATA_PATH,
                           batch_size=64,
                           data_split=[3, 1],
                           num_patients_train=num_patients_per_label,
                           num_patients_test=num_patients_per_label,
                           num_workers=args.num_workers
                           )
        dm.setup()
        sub_train_set = prepare_data_features(model, dm.train_dataloader(), device, **dict_args)
        test_feats_simclr = prepare_data_features(model, dm.test_dataloader(), device, **dict_args)

        _, small_set_results = train_logreg(train_feats_data=sub_train_set,
                                            test_feats_data=test_feats_simclr,
                                            model_suffix=num_patients_per_label,
                                            feature_dim=int(constants.SLEEP_EPOCH_SIZE/8 * args.model_hparams['conv_filters'][-1]),
                                            device=device,
                                            **dict_args)
        results[num_patients_per_label] = small_set_results


    dataset_sizes = sorted([k for k in results])
    test_scores = [results[k]["test"] for k in dataset_sizes]

    for k, score in zip(dataset_sizes, test_scores):
        print(f'Test accuracy for {k:3d} patients: {100 * score:4.2f}%')

    fig = plt.figure(figsize=(6, 4))
    plt.plot(dataset_sizes, test_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y",
             markersize=16)
    plt.xscale("log")
    plt.xticks(dataset_sizes, labels=dataset_sizes)
    plt.title("STL10 classification over dataset size", fontsize=14)
    plt.xlabel("Number of images per class")
    plt.ylabel("Test accuracy")
    plt.minorticks_off()
    plt.show()


def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, device, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(kwargs['CHECKPOINT_PATH'], "LogisticRegression"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=False,
                         check_val_every_n_epoch=10)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(kwargs['CHECKPOINT_PATH'], f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result

@torch.no_grad()
def prepare_data_features(model, data_loader, device, **kwargs):
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

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels)