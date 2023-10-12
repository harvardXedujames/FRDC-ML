from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, Subset


class FRDCDataModule(LightningDataModule):
    def __init__(
            self,
            *,
            segments: list[np.ndarray],
            fn_segment_tf: Callable[[list[np.ndarray]], torch.Tensor],
            labels: list[str] | None = None,
            fn_split: Callable[[TensorDataset], list[Subset, Subset, Subset]] =
                lambda x: [x, x, x],
            batch_size: int = 4,
    ):
        """ FRDC Data Module.

        Notes:
            We separate the segment input and transform as we expect the input
            to be as "raw" as possible, which is usually in the np.ndarray
            format.

            This makes it easy if we have custom images with alternative
            segmentations, such as through an auto-segmentation.

        Args:
            segments: A list of segments, each ND-Array.
                Note that PyTorch expects input shapes of (B, C, H, W),
                 while common image libs expect shapes (H, W, C).
            labels: A list of labels as strings. If None, then the datamodule
                will not have train, val, test datasets.
            fn_segment_tf: Transform applied to the segments.
                It takes a list of segments, returning a batched tensor:
                 (batch, height, width, channels).
                In FRDC, each segment is the tree crown segment, the output
                 should be a batched image of all the crown segments.
                See Examples for more details.
            fn_split: This is a function that takes a TensorDataset and splits
                it into train, val, test TensorDatasets.
                See Examples for more details.
            batch_size: The batch size to use for the dataloaders.

        Examples:
            The fn_segment_tf could be a function that resizes the segments to
            299x299 and then stacks them into a batched image. The output is
            then permuted to be (batch, channels, height, width).:

            >>> from skimage.transform import resize
            >>> from frdc.models import FaceNet
            >>>
            >>> fn_segment_tf=lambda x: torch.stack([
            >>>     torch.from_numpy(
            >>>         resize(s, [FaceNet.MIN_SIZE, FaceNet.MIN_SIZE])
            >>>     ) for s in x
            >>> ]).permute(0, 3, 1, 2)

            The fn_split could be a function that splits the dataset into
            train, val, test subsets.:

            >>> from torch.utils.data import random_split
            >>>
            >>> fn_split=lambda x: random_split(x, lengths=[len(x) - 6, 3, 3])

        """
        super().__init__()
        self.segments = segments
        self.labels = labels
        self.le = LabelEncoder()
        self.batch_size = batch_size
        self.fn_segment_tf = fn_segment_tf
        self.fn_split = fn_split

    def setup(self, stage=None):
        # Split dataset into train, val, test
        self.x = self.fn_segment_tf(self.segments)

        assert torch.isnan(self.x).sum() == 0, \
            "Found NaN values in the segments."
        assert self.x.ndim == 4, \
            (f"Expected 4 dimensions, got {self.x.ndim} dimensions of shape"
             f" {self.x.shape}.")

        if stage in ['fit', 'validate', 'test']:
            if self.labels is None:
                raise ValueError("Provide labels for fit, validate, or test.")

            self.y = torch.from_numpy(self.le.fit_transform(self.labels))
            assert self.x.shape[0] == self.y.shape[0], \
                (f"Expected same number of samples for x and y, got"
                 f" {self.x.shape[0]} for x and {self.y.shape[0]} for y.")

            tds = TensorDataset(self.x, self.y)
            self.train_ds, self.val_ds, self.test_ds = self.fn_split(tds)

        elif stage == 'predict':
            tds = TensorDataset(self.x)
            self.predict_ds = tds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size,
                          shuffle=False)
