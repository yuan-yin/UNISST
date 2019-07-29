import torch
import numpy as np
import collections
import matplotlib.pyplot as plt


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def size(input_, dim=0):
    """get batch size"""
    if isinstance(input_, torch.Tensor):
        return input_.shape[dim]
    elif isinstance(input_, collections.Mapping):
        return input_[next(iter(input_))].shape[dim]
    elif isinstance(input_, collections.Sequence):
        return input_[0].shape[dim]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}"
                         .format(input_type, type(input_))))

def colorize(tensor, vmin=-1, vmax=1, cmap='viridis'):
    vmin = tensor.min() if vmin is None else vmin
    vmax = tensor.max() if vmax is None else vmax
    tensor = (tensor - vmin) / (vmax - vmin)
    tensor = tensor.squeeze(1)
    ttype = tensor.dtype

    array = tensor.clone().detach().numpy()

    cmap = plt.get_cmap(cmap)

    array_rgba = cmap(array)
    array_rgb = np.delete(array_rgba, 3, axis=3)

    tensor_rgb = torch.from_numpy(array_rgb)
    tensor_rgb = tensor_rgb.permute(0, 3, 1, 2)
    tensor_rgb = tensor_rgb * (vmax - vmin)  + vmin
    return tensor_rgb.to(ttype)