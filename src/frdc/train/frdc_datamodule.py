from __future__ import annotations

from dataclasses import dataclass

from lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from frdc.load import FRDCDataset


@dataclass
class FRDCDataModule(LightningDataModule):
    """Lightning DataModule for FRDC Dataset.

    Notes:
        This is a special datamodule for semi-supervised learning, which
        requires two dataloaders for the labelled and unlabelled datasets.
        It can also be used for supervised learning, by passing in None for
        the unlabelled dataset.

        If you're using our MixMatch Module, using None for the unlabelled
        dataset will skip the MixMatch. However, note that this is not
        equivalent to passing the Labelled set as unlabelled as well.

        For example:
        >>> FRDCSSLDataModule(
        ...     train_lab_ds=train_lab_ds,
        ...     train_unl_ds=train_lab_ds,
        ...     ...
        ... )

        Does not have the same performance as:
        >>> FRDCSSLDataModule(
        ...     train_lab_ds=train_lab_ds,
        ...     train_unl_ds=None,
        ...     ...
        ... )

        As partially, some samples in MixMatch uses the unlabelled loss.

    Args:
        train_lab_ds: The labelled training dataset.
        train_unl_ds: The unlabelled training dataset. Can be None, which will
            default to a DataModule suitable for supervised learning.
        val_ds: The validation dataset.
        batch_size: The batch size to use for the dataloaders.
        train_iters: The number of iterations to run for the labelled training
            dataset.
        val_iters: The number of iterations to run for the validation dataset.

    """

    train_lab_ds: FRDCDataset
    val_ds: FRDCDataset
    train_unl_ds: FRDCDataset | None = None
    batch_size: int = 4
    train_iters: int = 100
    val_iters: int = 100

    def __post_init__(self):
        super().__init__()

    def train_dataloader(self):
        num_samples = self.batch_size * self.train_iters
        lab_dl = DataLoader(
            self.train_lab_ds,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                self.train_lab_ds,
                num_samples=num_samples,
                replacement=False,
            ),
        )
        unl_dl = (
            DataLoader(
                self.train_unl_ds,
                batch_size=self.batch_size,
                sampler=RandomSampler(
                    self.train_unl_ds,
                    num_samples=self.batch_size * self.train_iters,
                    replacement=False,
                ),
            )
            if self.train_unl_ds is not None
            # This is a hacky way to create an empty dataloader.
            # The size should be the same as the labelled dataloader so that
            #  the iterator doesn't prematurely stop.
            else DataLoader(
                empty := [[] for _ in range(len(self.train_lab_ds))],
                batch_size=self.batch_size,
                sampler=RandomSampler(
                    empty,
                    num_samples=num_samples,
                    replacement=False,
                ),
            )
        )

        return [lab_dl, unl_dl]

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
        )
