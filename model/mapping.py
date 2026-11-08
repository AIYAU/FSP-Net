import torch
import torch.nn as nn

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


class FADM(Mapping):
    """Domain mapping with a residual branch for low spectral frequencies."""

    def __init__(self, in_dimension, out_dimension, num_modes):
        super().__init__(in_dimension, out_dimension)
        if num_modes < 1:
            raise ValueError("num_modes must be positive")
        self.num_modes = num_modes
        self.frequency_features = nn.Sequential(
            nn.Conv2d(in_dimension, out_dimension, 1, bias=False),
            nn.BatchNorm2d(out_dimension),
            nn.GELU(),
        )
        self.frequency_projection = nn.Conv2d(out_dimension, out_dimension, 1, bias=False)
        self.frequency_bn = nn.BatchNorm2d(out_dimension)
        self.zero_frequency_projection()

    def zero_frequency_projection(self):
        nn.init.zeros_(self.frequency_projection.weight)

    def forward(self, x):
        base = super().forward(x)

        # FFT along the spectral bands, independently at each spatial location.
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spectrum = torch.fft.rfft(fft_input, dim=1)
        modes_to_keep = min(self.num_modes, spectrum.size(1))
        low_spectrum = torch.zeros_like(spectrum)
        low_spectrum[:, :modes_to_keep] = spectrum[:, :modes_to_keep]
        low_frequency = torch.fft.irfft(low_spectrum, n=x.size(1), dim=1).to(x.dtype)

        frequency = self.frequency_features(low_frequency)
        frequency = self.frequency_bn(self.frequency_projection(frequency))
        return base + frequency
