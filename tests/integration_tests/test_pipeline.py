import numpy as np

from frdc.evaluate import dummy_evaluate
from frdc.load import FRDCDownloader
from frdc.preprocess import compute_segments_mask, extract_segments
from frdc.train import dummy_train


def test_auto_segmentation_pipeline():
    """ Test the auto segmentation pipeline. This is used to preliminarily extract segments from the dataset. """

    ds = FRDCDownloader()._load_debug_dataset()
    ar = ds.ar_bands
    ar_segments_mask = compute_segments_mask(ar)
    ar_segments = extract_segments(ar, ar_segments_mask)


def test_manual_segmentation_pipeline():
    """ Test the manual segmentation pipeline. This is after we manually segment the dataset. """

    ds = FRDCDownloader()._load_debug_dataset()
    ar = ds.ar_bands
    # This is a trivial example of manual segmentation, which bins the first band into 4 segments with equal quantiles.
    # In production, this will be loaded from a ground truth mask.
    qs = [0, 0.25, 0.5, 0.75, 1]
    ar_segments_mask = np.digitize(ar[:, :, 0], np.nanquantile(ar[:, :, 0], qs), False) - 1
    ar_segments = extract_segments(ar, ar_segments_mask)

    # assert that the number of labeled pixels is equal to the number of pixels in the segments
    # ! We -1 because digitize implicitly adds a bin for values above the last quantile
    #   Thus the comparison will not be valid for the last segment
    # TODO: Probably get a ground truth mask from the dataset to avoid this hacky code.
    for segment_ix in range(len(qs) - 1):
        mask_pixels_segment = np.sum(ar_segments_mask == segment_ix)
        non_nan_pixels_segment = np.sum(~np.isnan(ar_segments[segment_ix]))
        assert mask_pixels_segment * ar.shape[-1] == non_nan_pixels_segment


def test_pipeline():
    """ Test the whole pipeline. """
    ds = FRDCDownloader()._load_debug_dataset()
    ar = ds.ar_bands
    ar_segments_mask = compute_segments_mask(ar)
    ar_segments = extract_segments(ar, ar_segments_mask)

    # 1: to skip the background
    X = np.stack(ar_segments[1:])

    # TODO: Randomly generate y for now.
    y = np.random.randint(0, 3, size=(X.shape[0]))

    # TODO: We'll need to be smart on how we split the data.
    X_train, X_val, X_test = X[:-6], X[-6:-3], X[-3:]
    y_train, y_val, y_test = y[:-6], y[-6:-3], y[-3:]

    feature_extraction, classifier, val_score = dummy_train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    test_score = dummy_evaluate(feature_extraction=feature_extraction, classifier=classifier,
                                X_test=X_test, y_test=y_test)

    print(f"Validation score: {val_score:.2%}")
    print(f"Test score: {test_score:.2%}")
