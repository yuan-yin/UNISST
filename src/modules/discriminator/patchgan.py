### Code from: https://raw.githubusercontent.com/NVIDIA/vid2vid/master/models/networks.py

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm

# Defines the PatchGAN discriminator with the specified arguments.
class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        use_bias = True

        kw = 3
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, mask=None):
        """Standard forward."""
        return self.model(input)

class Discriminator3d(nn.Module):
    """Defines a 3D-PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, pooling=False):

        super().__init__()
        use_bias = True
        self.pooling = pooling

        kw = (3,3,3)
        padw = (1,1,1)
        sequence = [spectral_norm(nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=(1,2,2), padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1,2,2), padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1,1,1), padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=(1,1,1), padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, mask=None):
        """Standard forward."""
        if self.pooling:
            return self.model(input).sum(dim=(2,3,4))
        else:
            return self.model(input)