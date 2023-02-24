import torch
from abc import ABC, abstractmethod
import constants
import scipy.signal as signal
import torch.nn as nn

class AugmentationModule(nn.Module):
    def __init__(self,
                 amplitude_min: int,
                 amplitude_max: int,
                 timeshift_min: int,
                 timeshift_max: int,
                 zeromask_min: int,
                 zeromask_max: int,
                 noise_min: float,
                 noise_max: float,
                 bandstop_min: int,
                 bandstop_max: int,
                 freq_window: int):
        super().__init__()
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.timeshift_min = timeshift_min
        self.timeshift_max = timeshift_max
        self.zeromask_min = zeromask_min
        self.zeromask_max = zeromask_max
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.bandstop_min = bandstop_min
        self.bandstop_max = bandstop_max
        self.freq_window = freq_window
    def forward(self, x):
        # Must work for a batch!!
        x = x * (self.amplitude_min + torch.rand(1)*(self.amplitude_max-self.amplitude_min))  # Amplitude scale



class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    # TODO: optimize this procedure
    # Idea: copy x once here, change x itself in all
    def __call__(self, x, prev_x):
        ylist = list()
        for i in range(self.n_views):
            y = torch.clone(x)
            for T in self.base_transforms:
                y = T(y, prev_x)
            ylist.append(y)
        return ylist


class TranformProb(ABC):
    """
        Abstract base class for transformations that are applied within a uniform range between min and max and with a probability p
    """

    def __init__(self, mini, maxi, prob, batch_size):
        self.min = mini
        self.max = maxi
        self.prob = prob
        self.batch_size = batch_size

    @abstractmethod
    def action(self, x, x_prev):
        pass

    def __call__(self, x, x_prev):
        self.u = self.min + torch.rand(self.batch_size) * (
                self.max - self.min)  # Uniform number between min and max (different for every sample)
        if torch.rand(1) < self.prob:
            x = self.action(x, x_prev)
        return x


class AmplitudeScale(TranformProb):
    """
        This class applies an amplitude scale (uniformly) between min and max with a probability p
    """

    def __init__(self, mini, maxi, prob, batch_size):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x, x_prev):
        return x * self.u.unsqueeze(1)


def roll_with_different_shifts(x, shifts):
    n_rows, n_cols = x.shape
    arange1 = torch.arange(n_cols).view((1, n_cols)).repeat(n_rows, 1)
    arange2 = (arange1 - shifts[..., None]) % n_cols
    return torch.gather(x, 1, arange2)


class TimeShift(TranformProb):
    """
        This class applies a time shift between min and max (uniformly) samples with a probability p
    """

    def __init__(self, mini, maxi, prob, batch_size):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x, x_prev):
        shifts = int(self.u)  # Uniform number between min and max
        # TODO: do a proper timeshift and try to implement it with tensor operations
        # y = roll_with_different_shifts(x, shifts)
        y = torch.zeros_like(x)
        y[..., shifts:] = x[..., :-shifts]
        y[..., :shifts] = x_prev[..., -shifts:]
        return y

class DCShift(TranformProb):
    """
        This class applies a DC shift uniformly between min and max with a probability p
    """

    def __init__(self, mini, maxi, prob, batch_size, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x, x_prev):
        return x + self.u[..., None]


class ZeroMask(TranformProb):
    """
        This class applies a zero mask of a length uniformly between min and max to a part of the signal with a probability p
    """

    def __init__(self, mini, maxi, prob, batch_size, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x, x_prev):
        u = int(self.u)
        start = int(constants.SLEEP_EPOCH_SIZE * torch.rand(1))
        y = x.detach().clone()
        if start > constants.SLEEP_EPOCH_SIZE - u:
            y[..., start:] = 0  # TODO: make this work for a batch!
        else:
            y[..., start:start+u] = 0
        return y


class GaussianNoise(TranformProb):
    """
        This class adds gaussian noise of stdev between min and max with a probability p
    """

    def __init__(self, mini, maxi, prob, batch_size):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x, x_prev):
        return x + self.u * torch.randn(constants.SLEEP_EPOCH_SIZE)


class BandStopFilter(TranformProb):
    """
        This class applies a band-stop filter (10 Hz width)
    """

    def __init__(self, mini, maxi, prob, batch_size, freq_window, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)
        self.freq_window = freq_window

    def action(self, x, x_prev):
        start_freq = int(self.u)
        sos = signal.butter(8, [start_freq, start_freq + self.freq_window], btype="bandstop", output="sos", fs=constants.fs)
        x_filtered = torch.as_tensor(signal.sosfilt(sos, x.squeeze()), dtype=torch.float32)
        x_filtered = x_filtered[None, None, :]
        return x_filtered
