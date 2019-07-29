import torch

from src.corruptions import init_corruption
from src.datasets import init_dataset
from src.experiments import init_experiment
from src.modules import init_module
from src.optimizers import init_optimizer
from src.utils.sacred import sacred_run


def init_and_run(experiment, modules, corruption, corruption_test, datasets, optimizers, _run, _log, _seed):
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    
    # initializing corruption
    corr = init_corruption(**corruption)
    corr_test = init_corruption(**corruption_test)

    # initializing datasets
    dsets = {}
    for dataset_name, dataset_config in datasets.items():
        if dataset_name in ['test', 'val']:
            dsets[dataset_name] = init_dataset(corr_test, **dataset_config)
        else:
            dsets[dataset_name] = init_dataset(corr, **dataset_config)

    # initializing modules
    mods = {}
    for module_name, module_config in modules.items():
        mods[module_name] = init_module(**module_config)

    # initializing optimizers
    optims = {}
    for optimizer_name, optimizer_config in optimizers.items():
        optims[optimizer_name] = init_optimizer(mods, **optimizer_config)

    # initializing experiment and running it
    init_experiment(sacred_run=_run, seed=_seed,
                    corruption=corr,
                    **dsets, **mods, **optims,
                    **experiment).run()


if __name__ == '__main__':
    sacred_run(init_and_run)
