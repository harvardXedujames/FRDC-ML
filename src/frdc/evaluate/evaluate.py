from typing import Callable

import numpy as np
from sklearn.base import ClassifierMixin


def dummy_evaluate(
    *,
    feature_extraction: Callable[[np.ndarray], np.ndarray],
    classifier: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Dummy Evaluation function.

    Notes:
        This is obviously not final. This is just a placeholder to get the
         pipeline working.

    Args:
        feature_extraction: The feature extraction function.
        classifier: The classifier.
        X_test: X_test is the test image numpy array of shape (N, H, W, C).
        y_test: y_test is the test class label a numpy array of shape (N,).

    Returns:
        The score of the model.
    """
    # TODO: Replace this with how the model scores

    return classifier.score(feature_extraction(X_test), y_test)
