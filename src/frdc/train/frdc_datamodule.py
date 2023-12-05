from __future__ import annotations

from dataclasses import dataclass

from lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from frdc.load import FRDCDataset


@dataclass
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

    def __post_init__(self):
        super().__init__()

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


# TODO: This is a superclass of FRDCDataModule, we can technically mould our
#       downstream tasks to accept the special train_dataloader(), which
#       returns a list of dataloaders instead of a single dataloader.
#       The above can be deprecated in favour of this.
@dataclass
class FRDCSSLDataModule(LightningDataModule):
    """FRDC Data Module.

    Args:
        batch_size: The batch size to use for the dataloaders.

    """

    train_lab_ds: FRDCDataset
    train_unl_ds: FRDCDataset
    val_ds: FRDCDataset
    batch_size: int = 4
    train_iters: int = 100
    val_iters: int = 100

    def __post_init__(self):
        super().__init__()

    def train_dataloader(self):
        return [
            DataLoader(
                self.train_lab_ds,
                batch_size=self.batch_size,
                sampler=RandomSampler(
                    self.train_lab_ds,
                    num_samples=self.batch_size * self.train_iters,
                    replacement=False,
                ),
            ),
            DataLoader(
                self.train_unl_ds,
                batch_size=self.batch_size,
                sampler=RandomSampler(
                    self.train_unl_ds,
                    num_samples=self.batch_size * self.train_iters,
                    replacement=False,
                ),
            ),
        ]

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
        )
