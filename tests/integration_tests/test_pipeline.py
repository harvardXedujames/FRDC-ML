import numpy as np
from sklearn.preprocessing import LabelEncoder

from frdc.evaluate import dummy_evaluate
from frdc.load import FRDCDataset
from frdc.preprocess import compute_labels, extract_segments_from_labels, extract_segments_from_bounds
from frdc.train import dummy_train


def test_auto_segmentation_pipeline(ds):
    """ Test the auto segmentation pipeline. This is used to preliminarily extract segments from the dataset. """

    ar = ds.get_ar_bands()
    ar_labels = compute_labels(ar)
    ar_segments = extract_segments_from_labels(ar, ar_labels)


def test_manual_segmentation_pipeline(ds):
    """ Test the whole pipeline. """
    ar = ds.get_ar_bands()
    ar = np.nan_to_num(ar)
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar, bounds, cropped=False)

    X = np.stack(segments)
    y = LabelEncoder().fit_transform(labels)

    # TODO: We'll need to be smart on how we split the data.
    X_train, X_val, X_test = X[:-6], X[-6:-3], X[-3:]
    y_train, y_val, y_test = y[:-6], y[-6:-3], y[-3:]

    feature_extraction, classifier, val_score = dummy_train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    test_score = dummy_evaluate(feature_extraction=feature_extraction, classifier=classifier,
                                X_test=X_test, y_test=y_test)

    print(f"Validation score: {val_score:.2%}")
    print(f"Test score: {test_score:.2%}")
