from src.experiments.unisst import UNISST


def get_experiment_by_name(name):
    if name == 'unisst':
        return UNISST
    raise NotImplementedError(name)

def init_experiment(_name, **kwargs):
    return get_experiment_by_name(_name)(**kwargs)
