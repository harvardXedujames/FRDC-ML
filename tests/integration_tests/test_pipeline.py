import logging

import numpy as np
from scipy.special import kl_div
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from frdc.evaluate import dummy_evaluate
from frdc.preprocess import compute_labels, extract_segments_from_labels, extract_segments_from_bounds
from frdc.train import dummy_train


def test_auto_segmentation_pipeline(ds):
    """ Tests the use case where we just want to automatically segment the image. """

    ar = ds.get_bands()
    ar_labels = compute_labels(ar)
    ar_segments = extract_segments_from_labels(ar, ar_labels)


def test_manual_segmentation_pipeline(ds):
    """ Test the use case where we manually segment the image, then train a model on it. """
    ar = ds.get_bands()
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

    logging.debug(f"Validation score: {val_score:.2%}")
    logging.debug(f"Test score: {test_score:.2%}")

    return feature_extraction, classifier


def test_unlabelled_prediction(ds):
    """ Test the use case where we have unlabelled data, and we want to predict the labels. """
    ar = ds.get_bands()
    bounds, labels = ds.get_bounds_and_labels()

    # TODO: We need to revisit how "unlabelled" is defined.
    unlabelled_bounds = [b for b, l in zip(bounds, labels) if l == 'Tree 1']
    unlabelled_segments = extract_segments_from_bounds(ar, unlabelled_bounds, cropped=False)
    X = np.stack(unlabelled_segments)
    feature_extraction, classifier = test_manual_segmentation_pipeline(ds)
    X = feature_extraction(X)
    y_pred = classifier.predict(X)


def test_consistency_sampling(ds):
    """ Test the use case where we want to sample most inconsistent segments from unlabeled data.

    In order to do this, we need to:
    1) Extract unlabelled segments
    2) Augment the unlabelled segments
    3) Predict the labels of the augmented segments
    4) Calculate the KL Divergence between the predictions
    5) Calculate the inconsistency of the KL Divergence
    6) Sample the most inconsistent segments

    """
    ar = ds.get_bands()
    bounds, labels = ds.get_bounds_and_labels()

    # TODO: We need to revisit how "unlabelled" is defined.
    unlabelled_bounds = [b for b, l in zip(bounds, labels) if l == 'Tree 1']
    unlabelled_segments = extract_segments_from_bounds(ar, unlabelled_bounds, cropped=False)

    X = np.stack(unlabelled_segments)
    feature_extraction, classifier = test_manual_segmentation_pipeline(ds)
    classifier: RandomForestClassifier

    # TODO: Add actual augmentations
    N_AUGMENTS = 10
    augments = [
        lambda x: x[np.random.choice(range(X.shape[0]), X.shape[0], replace=False)]
        for _ in range(N_AUGMENTS)
    ]

    y_pred_probas = []
    for augment in augments:
        X_features = feature_extraction(augment(X))
        y_pred_probas.append(classifier.predict_proba(X_features))

    y_pred_probas = np.stack(y_pred_probas)
    n_augs, n_labs, n_probs = y_pred_probas.shape

    # We want a KL Div between all the augmentations.
    # ar_kl: np.ndarray matrix of shape (n_labs, n_augs, n_augs)
    # E.g. ar_kl[2, 0, 4] measures the KL Divergence of the 3rd label between the 1st and 5th augmentations.
    ar_kl = np.zeros((n_labs, n_augs, n_augs))

    for label_ix in range(n_labs):
        for aug_i in range(n_augs):
            for aug_j in range(aug_i):
                kl = kl_div(y_pred_probas[aug_i, label_ix] + 1e-10,
                            y_pred_probas[aug_j, label_ix] + 1e-10)
                ar_kl[label_ix, aug_i, aug_j] = kl.sum()
                ar_kl[label_ix, aug_j, aug_i] = kl.sum()

    # TODO: Not sure how to calculate the inconsistency. We'll use the variance for now.
    ar_inconsistency = ar_kl.var(axis=(1, 2))

    # TODO: We sample the 3 most inconsistent segments.
    n_samples = 3
    ar_top_3_inconsistency = ar_inconsistency.argsort()[-n_samples:][::-1]

    print(ar_top_3_inconsistency)
