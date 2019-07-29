from functools import partial

import torch
import torch.nn as nn
from torch.nn import init

from src.utils.misc import pretty_wrap


def init_weights(net, name='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if name == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif name == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif name == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif name == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method {name} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def num_params(module):
    num_params = 0
    for param in module.parameters():
        num_params += param.numel()
    return num_params


def print_net(name, net, init_name, init_gain):
    s = f'Class: {net.__class__.__name__:}\n' \
        f'Init: {init_name}, Gain={init_gain}\n' \
        f'Number of parameters : {num_params(net) / 1e6:.3f}\n'
    print(pretty_wrap(title=name, text=s))


def get_z_random(batchSize: int, nz: int, device: str, random_type: str = 'gauss'):
    if random_type == 'uni':
        z = torch.rand(batchSize, nz, device=device) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batchSize, nz, device=device)
    elif random_type == 'gauss_conjugate':
        std = (torch.randn(batchSize, nz, device=device) * 0.5).exp()
        mean = torch.randn(batchSize, nz, device=device)
        z = torch.cat([mean, std], 1)
    return z
