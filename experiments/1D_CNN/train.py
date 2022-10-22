import pytorch_lightning as pl
import os
from task import CNNmodel
from data import EEGdataModule
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


DATASET_PATH = '../../data/'
CHECKPOINT_PATH = 'trained_models'
save_name = "model01"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

datamodule = EEGdataModule(DATASET_PATH, batch_size=64)
datamodule.setup()
pl.seed_everything(42)  # To be reproducable


trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
                     gpus=1 if str(device) == "cuda:0" else 0,
                     max_epochs=20,
                     callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                LearningRateMonitor("epoch")],
                     enable_progress_bar=True)

trainer.logger._log_graph = True
trainer.logger._default_hp_metric = None
# Check whether pretrained model exists. If yes, load it and skip training
pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")

model = CNNmodel(model_name="1D_CNN",
                 model_hparams={},
                 optimizer_name="Adam",
                 optimizer_hparams={
                     "lr": 1e-3,
                     "weight_decay": 1e-4
                 })
trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
# model = CNNmodel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

# Test best model on validation and test set
val_result = trainer.test(model, datamodule.val_dataloader(), verbose=False)
test_result = trainer.test(model, datamodule.train_dataloader(), verbose=False)
result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
