from .corruptions import Cloud


def init_corruption(_name, **kwargs):
    if _name == 'cloud':
        return Cloud(**kwargs)
    
    raise NotImplementedError(_name)
