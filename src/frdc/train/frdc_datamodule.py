from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from lightning import LightningDataModule
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

from frdc.load import FRDCDataset
from frdc.preprocess import extract_segments_from_bounds


class FRDCDataModule(LightningDataModule):
    def __init__(
            self,
            ds: FRDCDataset,
            ar_segment_transform: Callable[
                [list[np.ndarray]], torch.Tensor
            ],
            td_split: Callable[
                [TensorDataset], list[Subset, Subset, Subset]
            ],
            batch_size: int = 4,
    ):
        """

        Args:
            ds: The FRDCDataset to use
            ar_segment_transform: Transform applied to the segments.
                It takes a list of segments and return a batched tensor:
                 (batch, height, width, channels).
                In the case of FRDC each segment is the crown segment of each
                tree, the output should be a batched image of all the crown
                segments.
            td_split: This is a function that takes a TensorDataset and splits
                it into train, val, test TensorDatasets.
            batch_size: The batch size to use for the dataloaders.

        Examples:
            The ar_transform could be a function that resizes the segments to
            299x299 and then stacks them into a batched image:

            # >>> def ar_transform(segments):
            # >>>     segments = [resize(s, [299, 299]) for s in segments]
            # >>>     return np.stack(segments)
        """
        super().__init__()
        self.ds = ds
        self.ar_segment_transform = ar_segment_transform
        self.td_split = td_split
        self.le = LabelEncoder()
        self.batch_size = batch_size

    def prepare_data(self):
        ar, order = self.ds.get_ar_bands()
        bounds, labels = self.ds.get_bounds_and_labels()
        ar_segments = extract_segments_from_bounds(ar, bounds, cropped=False)
        # self.ar has the shape (batch, height, width, channels)
        self.t = self.ar_segment_transform(ar_segments)
        self.y = torch.from_numpy(self.le.fit_transform(labels))

    def setup(self, stage=None):
        # Split dataset into train, val, test
        tds = TensorDataset(self.t, self.y)
        self.train_ds, self.val_ds, self.test_ds = self.td_split(tds)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False)


