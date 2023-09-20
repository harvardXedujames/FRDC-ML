import numpy as np

from frdc.evaluate import dummy_evaluate
from frdc.load import FRDCDataset
from frdc.preprocess import compute_labels, extract_segments_from_labels, extract_segments_from_bounds
from frdc.train import dummy_train


def test_auto_segmentation_pipeline():
    """ Test the auto segmentation pipeline. This is used to preliminarily extract segments from the dataset. """

    ds = FRDCDataset._load_debug_dataset()
    ar = ds.get_ar_bands()
    ar_labels = compute_labels(ar)
    ar_segments = extract_segments_from_labels(ar, ar_labels)


def test_manual_segmentation_pipeline():
    """ Test the manual segmentation pipeline. This is after we manually segment the dataset. """

    ds = FRDCDataset._load_debug_dataset()
    ar = ds.get_ar_bands()
    bounds = ds.get_bounds()
    segments = extract_segments_from_bounds(ar, bounds, cropped=False)
    ar_segments = np.stack(segments, axis=-1)


def test_pipeline():
    """ Test the whole pipeline. """
    ds = FRDCDataset._load_debug_dataset()
    ar = ds.get_ar_bands()
    ar_labels = compute_labels(ar)
    ar_segments = extract_segments_from_labels(ar, ar_labels, cropped=False)

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
