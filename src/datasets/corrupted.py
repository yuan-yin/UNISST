import torch
from torch.utils.data import Dataset

class CorruptedSST(Dataset):
    def __init__(self, dataset, corruption, output_transform=lambda x: x):
        self.dataset = dataset
        self.corruption = corruption
        self.output_transform = output_transform
        self.thetas = dict()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        batch = self.dataset[item]
        x, mean, std = self.output_transform(batch)
        seq_len = x.size(1)

        theta = self.thetas.get(item)
        ys, masks = [], []
        for t in range(seq_len):
            y, mask, theta = self.corruption(x[:, t].unsqueeze(0), t, theta=theta, device='cpu', seed=item)
            ys.append(y.squeeze(0))
            masks.append(mask.squeeze(0))

        self.thetas[item] = theta

        return {
            'x': x,
            'y': torch.stack(ys, dim=1),
            'mask': torch.stack(masks, dim=1),
            'theta': theta,
            'seq_len': int(seq_len),
            'mean': mean,
            'std': std,
        }