"""
taken from https://github.com/mit-han-lab/torchquantum/blob/master/torchquantum/datasets/mnist.py
"""


import torch

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List
from torchpack.utils.logging import logger
from torchvision.transforms import InterpolationMode


__all__ = ['CIFAR']


resize_modes = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'nearest': InterpolationMode.NEAREST,
}


class CIFARDataset:
    def __init__(self,
                 root: str,
                 split: str,
                 train_valid_split_ratio: List[float],
                 center_crop,
                 resize,
                 resize_mode,
                 binarize,
                 binarize_threshold,
                 digits_of_interest,
                 n_test_samples,
                 n_valid_samples,
                 n_train_samples,
                 same_n_samples_each_class=False,
                 device='cpu'):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_modes[resize_mode]
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.digits_of_interest = digits_of_interest
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples
        self.n_train_samples = n_train_samples

        self.n_digits = len(digits_of_interest)

        self.same_n_samples_each_class = same_n_samples_each_class

        self.device=device
        self.load()
        self.n_instance = len(self.data)


    def _get_indices(self, subset, n_samples):
        sample_ctr = {}
        sample_quota = {}
        indices = []
        n_samples_each_class = n_samples // self.n_digits

        for digit_of_interest in self.digits_of_interest:
            sample_ctr[digit_of_interest] = 0
            if digit_of_interest == self.digits_of_interest[-1]:
                sample_quota[digit_of_interest] = \
                    n_samples - (self.n_digits - 1) * n_samples_each_class
            else:
                sample_quota[digit_of_interest] = n_samples_each_class

        for idx in subset.indices:
            digit = subset.dataset[idx][1]
            if sample_ctr[digit] < sample_quota[digit]:
                indices.append(idx)
                sample_ctr[digit] += 1

        return indices

    def load(self):
        tran = [transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize((0.1307,), (0.3081,))]
        if not self.center_crop == 28:
            tran.append(transforms.CenterCrop(self.center_crop))
        if self.resize == 28:
            tran.append(transforms.Resize(self.resize,
                                          interpolation=self.resize_mode))
        transform = transforms.Compose(tran)

        if self.split == 'train' or self.split == 'valid':

            print("LOADING CIFAR10")
            train_valid = datasets.CIFAR10(self.root, train=True,
                                        download=True, transform=transform)
            train_valid.targets = torch.tensor(train_valid.targets).detach().cpu() 

            idx, _ = torch.stack([train_valid.targets == number for number in
                                  self.digits_of_interest]).max(dim=0)
            train_valid.targets = train_valid.targets[idx]
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            self.device='cuda:0'
            train_valid.targets = train_valid.targets.to(self.device)
            train_subset, valid_subset = torch.utils.data.random_split(
                #train_valid, split, generator=torch.Generator().manual_seed(1))
                train_valid, split, generator=torch.Generator(self.device).manual_seed(1))

            if self.split == 'train':
                if self.n_train_samples is None:
                    # use all samples in train set
                    self.data = train_subset
                else:
                    if self.same_n_samples_each_class:
                        train_subset.indices = self._get_indices(
                            train_subset,
                            self.n_train_samples)
                    else:
                        train_subset.indices = train_subset.indices[
                                           :self.n_train_samples]
                    self.data = train_subset
                    logger.warning(f"Only use the front "
                                   f"{self.n_train_samples} images as "
                                   f"TRAIN set.")
            else:
                if self.n_valid_samples is None:
                    # use all samples in valid set
                    self.data = valid_subset
                else:
                    # use a subset of valid set, useful to speedup evo search
                    if self.same_n_samples_each_class:
                        valid_subset.indices = self._get_indices(
                            valid_subset,
                            self.n_valid_samples)
                    else:
                        valid_subset.indices = valid_subset.indices[
                                           :self.n_valid_samples]
                    self.data = valid_subset
                    logger.warning(f"Only use the front "
                                   f"{self.n_valid_samples} images as "
                                   f"VALID set.")

        else:
            test = datasets.CIFAR10(self.root, train=False,
                                    transform=transform)
            test.targets = torch.tensor(test.targets).detach().cpu()
            idx, _ = torch.stack([test.targets == number for number in
                                  self.digits_of_interest]).max(dim=0)
            test.targets = test.targets[idx]
            test.data = test.data[idx]
            if self.n_test_samples is None:
                # use all samples as test set
                self.data = test
            else:
                # use a subset as test set
                if self.same_n_samples_each_class:
                    sample_ctr = {}
                    sample_quota = {}
                    indices = []
                    n_samples_each_class = self.n_test_samples // self.n_digits

                    for digit_of_interest in self.digits_of_interest:
                        sample_ctr[digit_of_interest] = 0
                        if digit_of_interest == self.digits_of_interest[-1]:
                            sample_quota[digit_of_interest] = \
                                self.n_test_samples - (self.n_digits - 1) * n_samples_each_class
                        else:
                            sample_quota[digit_of_interest] = n_samples_each_class

                    for idx, target in enumerate(test.targets):
                        digit = target.item()
                        if sample_ctr[digit] < sample_quota[digit]:
                            indices.append(idx)
                            sample_ctr[digit] += 1

                    test.targets = test.targets[indices]
                    test.data = test.data[indices]
                else:
                    test.targets = test.targets[:self.n_test_samples]
                    test.data = test.data[:self.n_test_samples]
                test.targets = test.targets.to('cuda:0')
                #test.data = test.data.to('cuda:0')
                self.data = test
                logger.warning(f"Only use the front {self.n_test_samples} "
                               f"images as TEST set.")

    def __getitem__(self, index: int):
        img = self.data[index][0]
        if self.binarize:
            img = 1. * (img > self.binarize_threshold) + \
                  -1. * (img <= self.binarize_threshold)

        digit = self.digits_of_interest.index(self.data[index][1])
        instance = {'x': img.to(self.device), 'y': digit}

        return instance

    def __len__(self) -> int:
        return self.n_instance


class CIFAR(Dataset):
    def __init__(self,
                 root: str,
                 train_valid_split_ratio: List[float],
                 device: str,
                 center_crop=28,
                 resize=28,
                 resize_mode='bilinear',
                 binarize=False,
                 binarize_threshold=0.1307,
                 digits_of_interest=tuple(range(10)),
                 n_test_samples=None,
                 n_valid_samples=None,
                 n_train_samples=None,
                 ):
        self.root = root

        super().__init__({
            split: CIFARDataset(
                root=root,
                split=split,
                train_valid_split_ratio=train_valid_split_ratio,
                center_crop=center_crop,
                resize=resize,
                resize_mode=resize_mode,
                binarize=binarize,
                binarize_threshold=binarize_threshold,
                digits_of_interest=digits_of_interest,
                n_test_samples=n_test_samples,
                n_valid_samples=n_valid_samples,
                n_train_samples=n_train_samples,
                device=device)
            for split in ['train', 'valid', 'test']
        })


