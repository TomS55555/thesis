import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import constants
from models.simclr_model import SimCLR
from utils.helper_functions import prepare_data_module


def train_simclr(args, device):
    pl.seed_everything(42)  # To be reproducable

    data_module = prepare_data_module(args.DATA_PATH, **args.data_hparams)

    trainer = Trainer(
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
