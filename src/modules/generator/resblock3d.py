import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SpatialBatchNorm(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        _, nc, _, h, w = input.shape
        x = input.view(-1, nc, h, w)
        x = super().forward(x)
        return x.view(*list(input.shape))

class Block3d(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, downsample=False, bias=False):

        super().__init__()

        self.upsample = upsample
        self.downsample = downsample

        self.b1 = nn.BatchNorm3d(num_features=in_channels)
        self.b2 = nn.BatchNorm3d(num_features=out_channels)
        self.residual = in_channels != out_channels
        self.activation = nn.ReLU()

        self.c1 = spectral_norm(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias))
        self.c2 = spectral_norm(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias))
        if self.residual:
            self.c_sc = spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias))

    def forward(self, x):
        h = self.b1(x)
        h = self.activation(h)
        h = _sample(h, self.upsample, self.downsample)
        h = self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)

        x = _sample(x, self.upsample, self.downsample)
        if self.residual:
            x = self.c_sc(x)
        return h + x


def _sample(x, upsample=False, downsample=False):
    if upsample:
        return F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
    elif downsample:
        return F.avg_pool3d(x, (1, 2, 2))
    else:
        return x