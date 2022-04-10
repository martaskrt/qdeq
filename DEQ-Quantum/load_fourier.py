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
                 device
                 ):
        self.n_valid_samples = n_valid_samples
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.split = split
        self.n_samples = self.n_valid_samples + self.n_train_samples + self.n_test_samples

        def target_function(x):
            """Generate a truncated Fourier series, where the data gets re-scaled."""
            degree = 1  # degree of the target function
            scaling = 1  # scaling of the data
            coeffs = [0.15 + 0.15j]*degree  # coefficients of non-zero frequencies
            coeff0 = 0.1  # coefficient of zero frequency
            res = coeff0
            for idx, coeff in enumerate(coeffs):
                exponent = np.complex128(scaling * (idx+1) * x * 1j)
                conj_coeff = np.conjugate(coeff)
                res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
            return np.real(res)
        
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


