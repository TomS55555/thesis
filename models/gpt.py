import torch
import constants
import torch.nn as nn
from models.sleep_transformer import InnerTransformer
import pytorch_lightning as pl


class SleepGPT(pl.LightningModule):
    """
        This model takes as input batches of time-frequency images [batch x time x feat]. The model objective is to
        predict a masked time feature from the remaining feature vectors. An MLP can then be put on top of the flattened
        transformer output to predict sleep-stages.
    """
    def __init__(self, time_dim, freq_dim, dim_feedforward, num_heads, num_layers):
        super().__init__()
        self.net = InnerTransformer(
            inner_dim=time_dim,
            feat_dim=freq_dim,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_layers=num_layers
        )
        self.loss = nn.CrossEntropyLoss()  # TODO: maybe this should be a MSE loss

    def mask(self, x):
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # 1): Mask one feature at random
        masked_batch, masked_idxs, real_input = self.mask(inputs)
        # 2) Propagate the batch through the network
        output = self.net(masked_batch)
        # 3) Calculate the loss between the predicted feature vector and the real one
        predicted_input = output[masked_idxs]
        loss = self.loss(predicted_input, real_input)

        self.log()