import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import constants
from datasets.augmentations import *
from datasets.datamodules import EEGdataModule
from models.simclr_model import SimCLR


def train_simclr(args, device):
    pl.seed_everything(42)  # To be reproducable

    data_hparams = args.data_hparams

    p = data_hparams["transform-prob"]  # probability of applying the transforms
    contrast_transforms = ContrastiveTransformations(
        [
            AmplitudeScale(data_hparams["amplitude-min"], data_hparams["amplitude-max"], p, 1),
            GaussianNoise(data_hparams["noise-min"], data_hparams["noise-max"], p, 1),
            ZeroMask(data_hparams["zeromask-min"], data_hparams["zeromask-max"], p, 1),
            TimeShift(data_hparams["timeshift-min"], data_hparams["timeshift-max"], p, 1),
            BandStopFilter(data_hparams['bandstop-min'], data_hparams["bandstop-max"], p, 1,
                           data_hparams['freq-window'])
        ], n_views=2
    )
    data_module = EEGdataModule(
        data_path=args.DATA_PATH,
        transform=contrast_transforms,
        **data_hparams)
    data_module.setup()

    trainer = Trainer(
        precision="bf16",
        default_root_dir=os.path.join(args.CHECKPOINT_PATH, args.save_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,  # How many GPUs/CPUs to use
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5",
                            save_last=True),
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch")],  # Log learning rate every epoch
        enable_progress_bar=True,
        **args.trainer_hparams
    )

    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    encoder = constants.ENCODERS[args.encoder](**args.encoder_hparams)
    proj_head = constants.PROJECTION_HEADS[args.projection_head](**args.projection_head_hparams)

    model = SimCLR(encoder=encoder,
                   projector=proj_head,
                   optim_hparams=args.optim_hparams,
                   temperature=args.temperature)

    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    model = SimCLR.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, data_module.val_dataloader(), verbose=False)
    result = {"val": val_result[0]["val_acc_top1"]}
    return model, result
