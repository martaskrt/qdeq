"""
adapted from https://github.com/mit-han-lab/torchquantum/blob/master/torchquantum/datasets/mnist.py and Schuld et al 2021
"""


import torch

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List
from torchpack.utils.logging import logger

import numpy as np
np.random.seed(42)

class FourierDataset:
    def __init__(self,
                 split: str,
                 n_valid_samples,
                 n_train_samples,
                 n_test_samples,
                 device
                 ):
        self.n_valid_samples = n_valid_samples
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.split = split
        self.n_samples = self.n_valid_samples + self.n_train_samples + self.n_test_samples

        def target_function(x):
            ''' our target function, could be modified
                to e.g. implementation of a truncated Fourier
                series with certain frequencies and coefficients
            '''
            return 1/np.sqrt(2)*(np.sin(x) + np.cos(x))
        
        x = np.linspace(-6, 6, self.n_samples)
        self.y = torch.tensor([target_function(x_) for x_ in x], requires_grad=False).to(device)
        self.x = torch.tensor(x, requires_grad=False).to(device)
        idxs = np.arange(self.n_samples)
        np.random.shuffle(idxs)

        if self.split == "train":
            self.idxs = idxs[:self.n_train_samples]
           
        elif self.split == "valid":
            self.idxs = idxs[self.n_train_samples:self.n_train_samples+self.n_valid_samples]
        elif self.split == "test":
            self.idxs = idxs[self.n_train_samples+self.n_valid_samples:]
        self.x = self.x[self.idxs]
        self.y = self.y[self.idxs]
    def __getitem__(self, index: int):
        return {'x': self.x[index], 'y': self.y[index]}

    def __len__(self) -> int:
        return len(self.idxs)


class Fourier(Dataset):
    def __init__(self,
                 n_valid_samples=1000,
                 n_train_samples=100,
                 n_test_samples=100,
                 device='cpu'):

        super().__init__({
            split: FourierDataset(
                split=split,
                n_valid_samples=n_valid_samples,
                n_train_samples=n_train_samples,
                n_test_samples=n_test_samples,
                device=device)
            for split in ['train', 'valid', 'test']
        })


