import torch
from abc import ABC, abstractmethod
import constants
import scipy.signal as signal
import torch.nn as nn


class AugmentationModule:
    def __init__(self,
                 batch_size: int,
                 amplitude_min: int = 0.9,
                 amplitude_max: int = 1.1,
                 timeshift_min: int = -100,
                 timeshift_max: int = 100,
                 zeromask_min: int = 300,
                 zeromask_max: int = 500,
                 noise_min: float = 0.0,
                 noise_max: float = 0.3,
                 bandstop_min: int = 3,
                 bandstop_max: int = 45,
                 freq_window: int = 3):
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
        self.batch_size = batch_size

    def augment(self, x):
        # Must work for a batch!!

        # Amplitude scale
        rand_am = self.amplitude_min + torch.rand(self.batch_size).to(x) * (self.amplitude_max-self.amplitude_min)
        x = self.amplitude_scale(x, rand_am)

        # Zero mask
        rand_start_idxs = torch.randint(0, constants.SLEEP_EPOCH_SIZE-self.zeromask_max, (self.batch_size, 1)).to(x)
        rand_end_idxs = rand_start_idxs + torch.randint(self.zeromask_min, self.zeromask_max, (self.batch_size, 1)).to(x)
        ranges = torch.cat((rand_start_idxs, rand_end_idxs), dim=1).int()
        x = self.zero_mask(x, ranges)

        # Gaussian noise
        rand_stdevs = self.noise_min + torch.rand(self.batch_size).to(x) * (self.noise_max - self.noise_min)
        x = self.gaussian_noise(x, rand_stdevs)

        # Time shift
        shifts = torch.randint(low=self.timeshift_min,
                               high=self.timeshift_max,
                               size=(self.batch_size,)).to(x)
        shifts = tuple(shifts.int().tolist())
        x = self.time_shift(x, shifts)

        return x

    def amplitude_scale(self, x, rand_am):
        x = torch.matmul(torch.diag(rand_am), x)
        return x

    def zero_mask(self, x, ranges):
        indices = torch.arange(x.shape[1]).unsqueeze(0).to(x)
        mask = ((indices >= ranges[:, 0].unsqueeze(1)) & (indices < ranges[:, 1].unsqueeze(1)))
        x.masked_fill_(mask, 0)
        return x

    def gaussian_noise(self, x, rand_stdevs):
        zs = torch.randn_like(x).to(x)
        noise = torch.matmul(torch.diag(rand_stdevs), zs)
        x = x + noise
        return x

    def time_shift(self, x, shifts: tuple):
        # TODO: make this more efficient by only using part of prev and next inputs that is necessary
        prev_inputs = torch.roll(x, shifts=1, dims=0)
        next_inputs = torch.roll(x, shifts=-1, dims=0)
        inputs = torch.cat((prev_inputs, x, next_inputs), dim=1)
        time_shifted = torch.stack([torch.roll(inputs[i], shifts[i], dims=0) for i in range(inputs.shape[0])])
        x = time_shifted[:, x.shape[-1]:2*x.shape[-1]]
        return x

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
            y[..., start:start + u] = 0
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
        sos = signal.butter(8, [start_freq, start_freq + self.freq_window], btype="bandstop", output="sos",
                            fs=constants.fs)
        x_filtered = torch.as_tensor(signal.sosfilt(sos, x.squeeze()), dtype=torch.float32)
        x_filtered = x_filtered[None, None, :]
        return x_filtered
