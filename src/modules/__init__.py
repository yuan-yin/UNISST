import torch

from .discriminator.patchgan import Discriminator, Discriminator3d
from .generator.resnet3d import Generator3d
from .utils import init_weights, print_net


def get_module_by_name(name):
    if name == 'gen_3d':
        return Generator3d
    if name == 'dis_3d':
        return Discriminator3d
    if name == 'dis':
        return Discriminator
    raise NotImplementedError(name)

def init_module(_name, init_name=None, init_gain=None, gpu_id=[], **kwargs):
    """Only works for network modules"""

    module = get_module_by_name(_name)(**kwargs)
    if (init_name is not None) and (init_gain is not None):
        init_weights(module, init_name, init_gain)

    print_net(_name, module, init_name, init_gain)

    if isinstance(gpu_id, (list,)):
        gpu_id = list(range(len(gpu_id)))
    else:
        gpu_id = [0]

    if len(gpu_id) > 0:
        assert (torch.cuda.is_available())
        module = torch.nn.DataParallel(module, gpu_id).cuda()  # multi-GPUs

    return module
