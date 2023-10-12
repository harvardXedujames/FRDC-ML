import logging

import lightning as pl
import torch
from skimage.transform import resize
from torch.utils.data import random_split

from frdc.models import FaceNet
from frdc.preprocess import extract_segments_from_labels, \
    extract_segments_from_bounds
from frdc.train import FRDCDataModule, FRDCModule
from utils import get_labels


def fn_segment_tf(x):
    x = [resize(s, [FaceNet.MIN_SIZE, FaceNet.MIN_SIZE])
         for s in x]
    x = [torch.from_numpy(s) for s in x]
    x = torch.stack(x)
    x = torch.nan_to_num(x)
    x = x.permute(0, 3, 1, 2)
    return x


fn_split = lambda x: random_split(x, lengths=[len(x) - 6, 3, 3])

BATCH_SIZE = 3


def test_manual_segmentation_pipeline(ds) -> tuple[FRDCModule, FRDCDataModule]:
    """ Manually segment the image according to bounds.csv,
        then train a model on it. """

    ar, order = ds.get_ar_bands()
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar, bounds, cropped=True)

    dm = FRDCDataModule(
        segments=segments,
        labels=labels,
        fn_segment_tf=fn_segment_tf,
        fn_split=fn_split,
        batch_size=BATCH_SIZE
    )
    m = FRDCModule(model=FaceNet())

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(m, datamodule=dm)

    val_loss = trainer.validate(m, datamodule=dm)[0]['val_loss']
    test_loss = trainer.test(m, datamodule=dm)[0]['test_loss']

    logging.debug(f"Validation score: {val_loss:.2%}")
    logging.debug(f"Test score: {test_loss:.2%}")

    return m, dm


def test_auto_segmentation_pipeline(ds):
    """ Automatically segment the image, then use a model to predict. """

    # Auto segmentation
    ar, order = ds.get_ar_bands()
    ar_labels = get_labels(ar, order)
    segments_auto = extract_segments_from_labels(ar, ar_labels)

    # Get our model trained on the bounds.csv
    m, _ = test_manual_segmentation_pipeline(ds)

    # Construct our datamodule for prediction
    dm_auto = FRDCDataModule(
        segments=segments_auto,
        labels=None,  # Labels can be none if we just want predictions.
        fn_segment_tf=fn_segment_tf,
        fn_split=fn_split,
        batch_size=BATCH_SIZE
    )

    trainer = pl.Trainer(fast_dev_run=True)
    # The predictions have a shape of (N, C), where N is the number of
    # segments, and C is the number of classes.
    predictions = torch.concat(trainer.predict(m, datamodule=dm_auto))

    assert predictions.shape[0] == len(segments_auto), \
        "Expected the same number of predictions as segments."

    logging.debug(f"Predictions: {predictions}")
    logging.debug(f"Class Predictions: {torch.argmax(predictions, dim=1)}")
