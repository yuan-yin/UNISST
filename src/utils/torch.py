import torch


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
        raise TypeError(f'input must contain {input_type},' \
                            f' dicts or lists; found {type(input_)}')
