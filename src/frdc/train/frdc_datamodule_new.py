from __future__ import annotations

from dataclasses import dataclass, field

import torch
from lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
)

from frdc.load import FRDCDataset

ToTensor = Compose([ToImage(), ToDtype(torch.float32, scale=True)])


@dataclass  # (kw_only=True) # only available when we use Py3.10
class FRDCDataModule(LightningDataModule):
    """FRDC Data Module.

    Args:
        batch_size: The batch size to use for the dataloaders.

    """

    train_ds: FRDCDataset
    val_ds: FRDCDataset
    batch_size: int = 4
    train_iters: int = 100
    val_iters: int = 100

    le: LabelEncoder = field(init=False, default=LabelEncoder())

    def __post_init__(self):
        super().__init__()

    def setup(self, stage: str) -> None:
        # TODO: We'll figure out the test set later.
        #       Our dataset is way too small, even if we create one, it'll
        #       be too small to be useful.
        ...

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
        )
