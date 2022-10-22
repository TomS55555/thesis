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

test_path = "trained_models/model01/lightning_logs/version_2/checkpoints/epoch=13-step=1302.ckpt"

trainer = pl.Trainer(gpus=1 if str(device) == "cuda:0" else 0,
                     max_epochs=20,
                     callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                LearningRateMonitor("epoch")],
                     enable_progress_bar=True)

trainer.logger._log_graph = True
trainer.logger._default_hp_metric = None
# Check whether pretrained model exists. If yes, load it and skip training
# pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
if os.path.isfile(test_path):
    print(f"Found pretrained model at {test_path}, loading...")
    model = CNNmodel.load_from_checkpoint(test_path) # Automatically loads the model with the saved hyperparameters
else:
    print("Couldn't find pretrained model")
    exit(1)

# model = CNNmodel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training
# Test best model on validation and test set
val_result = trainer.test(model, datamodule.val_dataloader(), verbose=False)
test_result = trainer.test(model, datamodule.test_dataloader(), verbose=False)
result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
print(result)
