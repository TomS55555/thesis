import torch.nn as nn
import torch
import constants


class SimpleMLP(nn.Module):
    """
        Defines a simple MLP with one hidden layer
    """
    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=out_features)
        )

    def forward(self, x):
        return self.model(x)


class CNN_head(nn.Module):
    """
        Defines the CNN head
    """
    def __init__(self, conv_filters, representation_dim):
        super().__init__()
        self.model = nn.Sequential(
            CNN_block(constants.SLEEP_EPOCH_SIZE, 3, 1, conv_filters[0]),
            CNN_block(constants.SLEEP_EPOCH_SIZE/2, 3, conv_filters[0], conv_filters[1]),
            CNN_block(constants.SLEEP_EPOCH_SIZE/4, 3, conv_filters[1], conv_filters[2]),
            nn.Flatten(), # The result is of size int(constants.SLEEP_EPOCH_SIZE/8 * model_hparams["conv_filters"][-1])
            nn.Linear(int(constants.SLEEP_EPOCH_SIZE/8 * conv_filters[-1]), representation_dim)
        )

    def forward(self, x):
        return self.model(x)


class CNN_block(nn.Module):
    """
        One CNN block consists of a 1D (3) convolution, a Max pooling and a Batch normalization
    """
    def __init__(self, input_size, kernel_size, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding="same",
                      bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.net(x)