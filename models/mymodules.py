import torch.nn as nn
import torch

SLEEP_EPOCH_SIZE = 3000


class SimpleMLP(nn.Module):
    """
        Defines a simple MLP with one hidden layer
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
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
            CNN_block(3, 1, conv_filters[0]),
            CNN_block(3, conv_filters[0], conv_filters[1]),
            CNN_block(3, conv_filters[1], conv_filters[2]),
            nn.Flatten(),  # The result is of size int(constants.SLEEP_EPOCH_SIZE/8 * model_hparams["conv_filters"][-1])
            nn.Linear(int(SLEEP_EPOCH_SIZE/8 * conv_filters[-1]), representation_dim)
        )

    def forward(self, x):
        return self.model(x)


class CNN_decoder(nn.Module):
    """
        Inverse of CNN_head for generational purpose
    """
    def __init__(self, conv_filters, representation_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(representation_dim, int(SLEEP_EPOCH_SIZE/8 * conv_filters[0])),
            Inverse_CNN_block(3, 1, conv_filters[0]),
            Inverse_CNN_block(3, conv_filters[0], conv_filters[1]),
            Inverse_CNN_block(3, conv_filters[1], conv_filters[2]),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)


class Inverse_CNN_block(nn.Module):
    """
        The purpose of this block is to invert the operation of a regular convolution block
    """
    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.net(x)

class CNN_block(nn.Module):
    """
        One CNN block consists of a 1D (3) convolution, a Max pooling and a Batch normalization
    """
    def __init__(self, kernel_size, in_channels, out_channels):
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


class BIG_CNN_head(nn.Module):
    """
        This is a big convolutional net based on the paper 'A convolutional net for sleep-stage scoring
        from raw single-channel EEG' which reportedly got an accuracy of 87% on the SHHS dataset
        It uses the two previous epochs and the next one for temporal context

        Input: 12000 -> 6000 -> 3000 -> 1500 -> 750 -> 375
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),  # Output: 6000
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),  # Output: 3000
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),  # Output: 1500
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),  # Output: 750
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),  # Output: 375
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),  # Output: 188
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),  # Output: 94
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),  # Output: 47
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),  # Output: 24
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),  # Output: 12
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),  # Output: 6
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),  # Output: 3
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),  # Output: 768
            nn.Linear(768, 100)
        )

    def forward(self, x):
        return self.net(x)
