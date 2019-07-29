import random

import numpy as np
import torch


class Corruption(object):
    def sample_theta(self, im_shape, seed=None):
        raise NotImplementedError
    
    def __call__(self, x, t, device, theta=None, seed=None):
        raise NotImplementedError

class Cloud(Corruption):
    def __init__(self, root, im_size=(64, 64), time_steps=10800, threshold=0.08, interval=10):
        super().__init__()
        self.cloud = np.load(root)
        self.time_steps = time_steps
        self.threshold = threshold
        self.interval = interval

    def sample_theta(self, batch_size, seed=None):
        s = np.random.RandomState(seed)
        start_t = s.randint(low=100, high=self.time_steps - 24 * (self.interval), size=(batch_size, 1))
        return {
            'start_t': start_t
        }

    def __call__(self, x, t, device, theta=None, real_seq_len=None, seed=None, mask=None):

        fx = x.clone()
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.uint8)
            batch_size, _, _, width = x.shape

            if t < 0:
                return fx, mask, theta

            if theta is None:
                if t == 0:
                    theta = self.sample_theta(batch_size=batch_size, seed=seed)
                elif t > 0:
                    raise ValueError('theta is required for all steps other than the initial one.')

            cloud = torch.tensor(self.cloud[theta['start_t'] + t * (self.interval)])
            mask = cloud > self.threshold
            
        fx[mask] = 1

        return fx, mask, theta