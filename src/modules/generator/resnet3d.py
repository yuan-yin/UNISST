import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from collections import OrderedDict

from .resblock3d import Block3d
from ..non_local import SelfAttention

class Generator3d(nn.Module):
    def __init__(self, ngf, input_nc=1, output_nc=1, use_bias=True):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        self.blocks = nn.ModuleDict(
            OrderedDict({
                'block1'   : Block3d(input_nc, ngf     ),
                'block2'   : Block3d(ngf     , ngf * 16),
                'block3'   : Block3d(ngf * 16, ngf *  8), 
                'block4'   : Block3d(ngf *  8, ngf *  4),
                'block5'   : Block3d(ngf *  4, ngf *  2),
                'attention': SelfAttention(ngf * 2),
                'block6'   : Block3d(ngf *  2, ngf     ),
                'bn'       : nn.BatchNorm3d(ngf),
                'relu'     : nn.ReLU(),
                'conv'     : spectral_norm(nn.Conv3d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)),
                'tanh'     : nn.Tanh(),
            })
        )

    def forward(self, x, mask=None):
        for name, module in self.blocks.items():
            if name == 'attention':
                batch_size, nc, t, h, w = x.shape
                x = x.view(-1, nc, h, w)
                x = module(x)
                x = x.view(batch_size, nc, t, h, w)
            else:
                x = module(x)
        return x