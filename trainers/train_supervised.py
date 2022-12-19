import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import constants
from datasets.datamodules import EEGdataModule
from models.supervised_model import SupervisedModel


def train_supervised(args, device, pretrained_encoder=None, pretrained_classifier=None, dm=None):
    if dm is None:
        data_module = EEGdataModule(data_path=args.data_path, **args.data_hparams)
        data_module.setup()
    else:
        data_module = dm

    trainer = Trainer(
        default_root_dir=os.path.join(args.CHECKPOINT_PATH, args.save_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
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

    model = SupervisedModel(encoder, classifier, args.optim_hparams)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    model = SupervisedModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    return model
