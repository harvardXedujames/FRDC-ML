""" Tests for the FaceNet model.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset. The test accuracy is then reported into the logs.
"""

import logging

import lightning as pl
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import random_split

from frdc.conf import BAND_CONFIG
from frdc.load import FRDCDataset
from frdc.models import FaceNet
from frdc.preprocess import (
    extract_segments_from_bounds, scale_static_per_band
)
# from frdc.preprocess.glcm_padded import glcm_padded
from frdc.train import FRDCDataModule, FRDCModule


# See FRDCDataModule for fn_segment_tf and fn_split
def fn_segment_tf(x):
    def per_segment_tf(s):
        s = scale_static_per_band(s, list(BAND_CONFIG.keys()))

        # Call the glcm_padded_cached if you want to cache the GLCM
        s_glcm = glcm_padded(s, step_size=7, bin_from=1, bin_to=8, radius=3)

        s = np.concatenate([s[..., np.newaxis], s_glcm], axis=-1)
        s = s.reshape(*s.shape[:2], -1)
        s = resize(s, [FaceNet.MIN_SIZE, FaceNet.MIN_SIZE])
        s = torch.from_numpy(s)
        return s

    x = torch.stack([per_segment_tf(xi) for xi in x])
    x = torch.nan_to_num(x)
    x = x.permute(0, 3, 1, 2)
    return x


def fn_split(x):
    return random_split(x, lengths=[len(x) - 3, 3, 0])


BATCH_SIZE = 5


def get_dataset(site, date, version):
    ds = FRDCDataset(site=site, date=date, version=version)
    ar, order = ds.get_ar_bands()
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar, bounds, cropped=True)
    return segments, labels


def test_facenet():
    # Retrieve the 20201218 dataset
    segments_0, labels_0 = get_dataset(
        'chestnut_nature_park', '20201218', None
    )
    dm_0 = FRDCDataModule(
        segments=segments_0,
        labels=labels_0,
        fn_segment_tf=fn_segment_tf,
        fn_split=fn_split,
        batch_size=BATCH_SIZE
    )

    m = FRDCModule(
        model=FaceNet(n_in_channels=64, n_out_classes=len(set(labels_0)))
    )

    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(m, datamodule=dm_0)

    # Retrieve the 20210510 dataset
    segments_1, labels_1 = get_dataset(
        'chestnut_nature_park', '20210510',
        '90deg43m85pct255deg/map'
    )
    dm_1 = FRDCDataModule(
        segments=segments_1,
        labels=None,
        fn_segment_tf=fn_segment_tf,
        fn_split=fn_split,
        batch_size=BATCH_SIZE
    )

    labels_1_pred = dm_0.le.inverse_transform(
        torch.argmax(
            torch.concat(
                trainer.predict(m, datamodule=dm_1), dim=0
            ), dim=1
        )
    )

    # Calculate the test accuracy
    test_acc = sum(labels_1 == labels_1_pred) / len(labels_1)

    logging.info(f"Test accuracy: {test_acc:.2%}")
