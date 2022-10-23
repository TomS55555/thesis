import torch
from abc import ABC, abstractmethod
import constants


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class TranformProb(ABC):
    """
        Abstract base class for transformations that are applied within a uniform range between min and max and with a probability p
    """
    def __init__(self, mini, maxi, prob, batch_size):
        self.min = mini
        self.max = maxi
        self.prob = prob
        self.batch_size = batch_size
        self.u = self.min + torch.rand(batch_size) * (self.max - self.min)  # Uniform number between min and max (different for every sample)

    @abstractmethod
    def action(self, x):
        pass

    def __call__(self, x):
        if torch.rand(1) < self.prob:
            x = self.action(x)
        return x


class AmplitudeScale(TranformProb):
    """
        This class applies an amplitude scale (uniformly) between min and max with a probability p
    """
    def __init__(self, mini, maxi, prob, batch_size, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x):
        x *= self.u[..., None]  # Broadcast to correct dimension
        return x


class TimeShift(TranformProb):
    """
        This class applies a time shift between min and max (uniformly) samples with a probability p
    """
    def __init__(self, mini, maxi, prob, batch_size, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x):
        shift = int(self.u)  # Uniform number between min and max
        # TODO: do a timeshift
        return x


class DCShift(TranformProb):
    """
        This class applies a DC shift uniformly between min and max with a probability p
    """
    def __init__(self, mini, maxi, prob, batch_size, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x):
        x += self.u[..., None]  # Broadcast to correct dimension
        return x


class ZeroMask(TranformProb):
    """
        This class applies a zero mask of a length uniformly between min and max to a part of the signal with a probability p
    """
    def __init__(self, mini, maxi, prob, batch_size, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x):
        u = int(self.u)
        start = int(constants.SLEEP_EPOCH_SIZE * torch.rand(1))
        if start > constants.SLEEP_EPOCH_SIZE - u:
            x[start:] = 0  # TODO: make this work for a batch!


class GaussianNoise(TranformProb):
    """
        This class adds gaussian noise of stdev between min and max with a probability p
    """
    def __init__(self, mini, maxi, prob, batch_size, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x):
        x += self.u * torch.randn(self.batch_size)


class BandStopFilter(TranformProb):
    """
        This class applies a band-stop filter (10 Hz width)
    """
    def __init__(self, mini, maxi, prob, batch_size, **kwargs):
        super().__init__(mini, maxi, prob, batch_size)

    def action(self, x):
        pass
