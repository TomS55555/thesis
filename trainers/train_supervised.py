import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import constants
from datasets.datamodules import EEGdataModule
from models.supervised_model import SupervisedModel


def train_supervised(args,
                     device,
                     dm: pl.LightningDataModule,
                     pretrained_encoder=None,
                     pretrained_classifier=None,
                     backbone=None):
    trainer = Trainer(
        default_root_dir=os.path.join(args.CHECKPOINT_PATH, args.save_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        reload_dataloaders_every_n_epochs=1 if dm.num_ds > 1 else 0,
        # Reload dataloaders to get different part of the big dataset
        devices=1,  # How many GPUs/CPUs to use
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
        **args.trainer_hparams
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    encoder = constants.ENCODERS[args.encoder](
        **args.encoder_hparams) if pretrained_encoder is None else pretrained_encoder

    classifier = constants.CLASSIFIERS[args.classifier](
        **args.classifier_hparams) if pretrained_classifier is None else pretrained_classifier

    model = SupervisedModel(encoder, classifier, args.optim_hparams, backbone)
    trainer.fit(model, datamodule=dm)

    model = SupervisedModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    return model
