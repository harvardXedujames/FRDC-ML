from typing import Callable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


# We force keyword arguments to make it easier to read the function signature.
def dummy_train(
    *, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[Callable[[np.ndarray], np.ndarray], ClassifierMixin, float]:
    """Dummy Training function.

    Notes:
        This is obviously not final. This is just a placeholder to get the
        pipeline working.

    Args:
        X_train: X_train is the train image numpy array of shape (N, H, W, C).
        y_train: y_train is the train class label a numpy array of shape (N,).
        X_val: X_val is the validation for X_train
        y_val: y_val is the validation for y_train

    Returns:
        The feature extraction function, the classifier, and validation score.

    """

    # TODO: Placeholder feature extraction for an image.
    #  We'll probably sub with a CNN later.
    def feature_extraction(X):
        return np.stack(
            [
                np.nanmean(X, axis=(1, 2, 3)),
                np.nanstd(X, axis=(1, 2, 3)),
            ],
            axis=-1,
        )

    X_train = feature_extraction(X_train)
    X_val = feature_extraction(X_val)

    # TODO: This is likely the last layer of the CNN.
    classifier = RandomForestClassifier()

    # TODO: "Train" the model.
    classifier.fit(X_train, y_train)

    # TODO: "Evaluate" the model.
    score = classifier.score(X_val, y_val)

    return feature_extraction, classifier, score
