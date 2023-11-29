from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
from torchvision.transforms.v2 import (
    RandomCrop,
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToImage,
    ToDtype,
    Resize,
)

from frdc.load import FRDCDataset
from frdc.load.dataset import FRDCConcatDataset

ToTensor = Compose([ToImage(), ToDtype(torch.float32, scale=True)])


@dataclass
class Transforms:
    train_tf: Callable[[np.ndarray], Any] = ToTensor
    val_tf: Callable[[np.ndarray], Any] = ToTensor
    test_tf: Callable[[np.ndarray], Any] = ToTensor


class DatasetTransform(Dataset):
    def __init__(self, ds, transform, target_transform):
        self.transform = transform
        self.target_transform = target_transform
        self.ds = ds

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return self.transform(x), self.target_transform([y])


@dataclass  # (kw_only=True) # only available when we use Py3.10
class FRDCDataModule(LightningDataModule):
    """FRDC Data Module.

    Notes:
        The transform can be very flexible, as long as it takes in a
        np.ndarray and returns a tensor.

        Take for example, in our SSL experiments, we can transform a single
        sample to a set of augmented samples.

        >>> transforms = Transforms(
        >>>     train_tf=lambda x: (tf1(x), tf2(x), tf3(x)),
        >>>     val_tf=lambda x: tf1(x),
        >>>     test_tf=lambda x: tf1(x),
        >>> )

        Just note that if you transform like so, you need to handle the
        batch downstream differently.

    Args:
        ds: The FRDCDataset to use.
        transforms: Transforms applied separately to train, val, test.
        batch_size: The batch size to use for the dataloaders.

    """

    ds: FRDCDataset | FRDCConcatDataset
    transforms: Transforms
    batch_size: int = 4
    train_iters: int = 100
    val_iters: int = 100

    le: LabelEncoder = field(init=False, default=LabelEncoder())
    train_ds: Dataset = field(init=False, default=None)
    val_ds: Dataset = field(init=False, default=None)

    # train_val_test_split: (
    #         Callable[[Dataset], Collection[Dataset, Dataset, Dataset]] | None
    # )

    # test_ds: Dataset = field(init=False, default=None)
    # predict_ds: Dataset = field(init=False, default=None)

    def __post_init__(self):
        super().__init__()

    def setup(self, stage: str) -> None:
        self.le.fit(self.ds.targets)
        # TODO: We'll figure out the test set later.
        #       Our dataset is way too small, even if we create one, it'll
        #       be too small to be useful.

        train_ix, val_ix = train_test_split(
            np.arange(len(self.ds)), test_size=0.1, random_state=42
        )
        self.train_ds = Subset(
            DatasetTransform(
                self.ds,
                self.transforms.train_tf,
                self.le.transform,
            ),
            train_ix,
        )
        self.val_ds = Subset(
            DatasetTransform(
                self.ds, self.transforms.val_tf, self.le.transform
            ),
            val_ix,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                self.train_ds,
                num_samples=self.batch_size * self.train_iters,
                replacement=False,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                self.val_ds,
                num_samples=self.batch_size * self.val_iters,
                replacement=False,
            ),
        )
