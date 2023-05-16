import torch
from abc import ABC, abstractmethod
import constants
from scipy import signal
import torch.nn as nn
import torch.fft
import numpy as np
import torchaudio.functional as F


class AugmentationModule(nn.Module):
    def __init__(self,
                 batch_size: int,
                 amplitude_min: float = 0.9,
                 amplitude_max: float = 1.1,
                 timeshift_min: int = -100,
                 timeshift_max: int = 100,
                 zeromask_min: int = 300,
                 zeromask_max: int = 500,
                 noise_min: float = 0.0,
                 noise_max: float = 0.3,
                 bandstop_min: int = 3,
                 bandstop_max: int = 45,
                 freq_window: int = 10):
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

        # The a and b coefficients need to be made on CPU, therefore make them already
        self.Nab = 100
        self.a_list = torch.zeros(self.Nab, dtype=torch.float32)
        self.b_list = torch.zeros(self.Nab, dtype=torch.float32)
        for i in range(self.Nab):
            start_freq = (torch.rand(1) * (50 - self.freq_window - 1)) + 0.1  # make sure 0 and end are never hit
            b, a = signal.butter(2, (start_freq, start_freq + self.freq_window), btype='bandstop', fs=100, output='ba')
            self.a_list[i] = a
            self.b_list[i] = b

    def forward(self, x):
        """
            x: size B x Epoch length
        """
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

        # Band pass
        rand_idx = int(torch.randint(low=0, high=self.Nab-1, size=(1,)))
        x = self.bandpass_filter(x, rand_idx)

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

    def freq_mask(self, x, ranges):
        masked_fft = self.zero_mask(torch.fft.rfft(x), ranges)
        return torch.fft.irfft(masked_fft)

    def bandpass_filter(self, x, rand_idx, fs=100):
        # Cutoff in Hz
        return F.lfilter(x, torch.as_tensor(self.a_list[rand_idx], dtype=torch.float32), torch.as_tensor(self.b_list[rand_idx], dtype=torch.float32), clamp=False)
        # x_filtered = torch.zeros_like(x).cpu()
        # start_freqs.cpu()
        # x.to("cpu")
        # for i in range(self.batch_size):
        #     sos = signal.butter(8, [start_freqs[i], start_freqs[i] + self.freq_window/2], btype="bandstop", output="sos", fs=fs)
        #     x_filtered[i, :] = torch.as_tensor(signal.sosfilt(sos, x.cpu()[i, :]), dtype=torch.float32)
        # return x_filtered.to("cuda:0")



class AugmentationModuleSTFT(nn.Module):
    """
        This module does augmentations on tensors of the form [batch x time x feat] where time x feat is a time-frequency image
    """
    def __init__(self,
                 batch_size: int,
                 amplitude_min: int = 0.9,
                 amplitude_max: int = 1.1,
                 timeshift_min: int = -100,
                 timeshift_max: int = 100,
                 time_mask_window: int = 3,
                 freq_mask_window: int = 3,
                 noise: float = 0.01):
        super().__init__()
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.timeshift_min = timeshift_min
        self.timeshift_max = timeshift_max

        self.time_mask_window = time_mask_window
        self.freq_mask_window = freq_mask_window

        self.batch_size = batch_size
        self.noise = noise

    def forward(self, x):
        # Must work for a batch!!

        # Amplitude scale
        rand_am = self.amplitude_min + torch.rand(self.batch_size).to(x.device) * (self.amplitude_max-self.amplitude_min)
        x = self.amplitude_scale(x, rand_am)

        # Zero mask time
        x = self.zero_mask_time(x)
        # Zero mask freq
        x = self.zero_mask_freq(x)

        # Gaussian noise
        x = self.gaussian_noise(x, self.noise)  #TODO: Look at MIT paper for good noise value

        # Time shift
        # x = self.time_shift(x, tuple(torch.randint(self.timeshift_min, self.timeshift_max, (self.batch_size, ))))

        return x

    def amplitude_scale(self, x, rand_am):
        batch_dim, time_dim, feat_dim = x.shape
        x = torch.bmm(x, rand_am.reshape(-1, 1, 1).to(x.device) * torch.eye(feat_dim, feat_dim).repeat(batch_dim, 1, 1).to(x.device))
        return x

    def zero_mask_time(self, x):
        batch_dim, time_dim, freq_dim = x.shape
        rows_to_zero = torch.randint(low=0, high=time_dim - self.time_mask_window, size=(batch_dim,)).to(x.device)
        mask = torch.ones((batch_dim, time_dim, freq_dim), device=x.device)
        index1 = rows_to_zero.unsqueeze(1) + torch.arange(self.time_mask_window, device=x.device)
        mask[torch.arange(batch_dim, device=x.device).unsqueeze(1).long(), index1.long(), :] = 0
        return mask * x

    def zero_mask_freq(self, x):
        batch_dim, time_dim, freq_dim = x.shape
        cols_to_zero = torch.randint(low=0, high=freq_dim - self.freq_mask_window, size=(batch_dim,)).to(x.device)
        mask = torch.ones((batch_dim, time_dim, freq_dim)).to(x.device)
        index2 = cols_to_zero.unsqueeze(1) + torch.arange(self.freq_mask_window, device=x.device)
        mask[torch.arange(batch_dim, device=x.device).unsqueeze(1).long(), :, index2.long()] = 0
        return mask * x

    def gaussian_noise(self, x, stdev):
        # Maybe look at pytorch for images for better noise
        zs = torch.randn_like(x).to(x.device)
        return x + zs*stdev

    def time_shift(self, x, shifts: tuple):
        # TODO: make this more resilient because of the rolling
        time_shifted = torch.stack([torch.roll(x[i], shifts[i], dims=1) for i in range(x.shape[0])])  # Roll along time axis
        return time_shifted
