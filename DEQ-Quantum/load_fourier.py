"""
adapted from https://github.com/mit-han-lab/torchquantum/blob/master/torchquantum/datasets/mnist.py and Schuld et al 2021
"""


import torch

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List
from torchpack.utils.logging import logger


class MNISTDataset:
    def __init__(self,
                 split: str,
                 n_valid_samples,
                 n_train_samples,
                 ):
        self.train_valid_split_ratio = train_valid_split_ratio
        self.n_valid_samples = n_valid_samples
        self.n_train_samples = n_train_samples

        x = np.linspace(-6, 6, 70)
        y = torch.tensor([target_function(x_) for x_ in x], requires_grad=False).to(device)
        x = torch.tensor(x, requires_grad=False).to(device)

    def target_function(self, x):
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
    def __getitem__(self, index: int):
        img = self.data[index][0]

        digit = self.digits_of_interest.index(self.data[index][1])
        instance = {'x': img, 'y': digit}

        return instance

    def __len__(self) -> int:
        return self.n_instance


class MNIST(Dataset):
    def __init__(self,
                 n_valid_samples=None,
                 n_train_samples=None,
                 ):

        super().__init__({
            split: MNISTDataset(
                split=split,
                n_valid_samples=n_valid_samples,
                n_train_samples=n_train_samples,
            )
            for split in ['train', 'valid']
        })


