import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.SHHS_dataset_timeonly import EEGdataModule
from models.conv_model_simple import CNNmodel_supervised_simple


def train_cnn_supervised_simple(args, device):
    save_name = args.model_name
    dict_args = vars(args)
    pl.seed_everything(42)  # To be reproducable

    data_module = EEGdataModule(**dict_args['data_hparams'])
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

    model = CNNmodel_supervised_simple(**dict_args)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    model = CNNmodel_supervised_simple.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, data_module.val_dataloader(), verbose=False)
    test_result = trainer.test(model, data_module.test_dataloader(), verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    return model, result