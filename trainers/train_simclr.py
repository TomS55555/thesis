import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.SHHS_dataset_timeonly import EEGdataModule
from datasets.augmentations import ContrastiveTransformations, AmplitudeScale, TimeShift, ZeroMask, GaussianNoise, BandStopFilter, DCShift
from models.simclr_model import CNNmodel_SimCLR


def train_simclr(args, device):
    save_name = args.model_name
    dict_args = vars(args)
    pl.seed_everything(42)  # To be reproducable

    data_hparams = dict_args['data_hparams']

    p = data_hparams["transform-prob"]  # probability of applying the transforms
    contrast_transforms = ContrastiveTransformations(
        [
            AmplitudeScale(data_hparams["amplitude-min"], data_hparams["amplitude-max"], p, 1),
            GaussianNoise(data_hparams["noise-min"], data_hparams["noise-max"], p, 1),
            ZeroMask(data_hparams["zeromask-min"], data_hparams["zeromask-max"], p, 1),
            TimeShift(data_hparams["timeshift-min"], data_hparams["timeshift-max"], p, 1),
            BandStopFilter(data_hparams['bandstop-min'], data_hparams["bandstop-max"], p, 1, data_hparams['freq-window'])
        ], n_views=2
    )
    data_module = EEGdataModule(transform=contrast_transforms, **data_hparams)
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

    model = CNNmodel_SimCLR(**dict_args)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    model = CNNmodel_SimCLR.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, data_module.val_dataloader(), verbose=False)
    test_result = trainer.test(model, data_module.test_dataloader(), verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    return model, result
