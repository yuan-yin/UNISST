import glob, os, zipfile, re
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import imageio as io

from scipy.io import netcdf

class SST(Dataset):
    def __init__(self, root:str, group:str='train', output_size=(64, 64), interval=1, **kwargs):
        self.output_width  = output_size[0]
        self.output_height = output_size[1]
        self.root  = root

        self.nc = netcdf.netcdf_file(root)
        self.thetao = self.nc.variables['thetao'].data.copy()

        n_frames = self.thetao.shape[0]

        indices = np.concatenate([np.arange(i, n_frames, interval) for i in range(interval)], axis=0)
        self.thetao = self.thetao.take(indices, axis=0)
        _, _, h, w = self.thetao.shape

        self.thetao = self.thetao.reshape(-1, 1, 24, h, w)
        self.thetao[self.thetao == -32767] = 0
        self.total = self.thetao.shape[0]
        self.group = group

        if group == 'train':
            self.thetao = self.thetao[:300]
        elif group == 'val':
            self.thetao = self.thetao[300:]
        elif group == 'test':
            self.thetao = self.thetao[:60]

        self.total = self.thetao.shape[0]

        self.thetao = self.thetao[:, :, :, :output_size[1], :output_size[0]]

    def __getitem__(self, index):
        item = torch.tensor(self.thetao[index].astype(float), dtype=torch.float)
        i_mean = item.mean()
        i_std = item.std()
        return (item - i_mean) / (i_std * 3), i_mean, i_std

    def __len__(self):
        return self.total
