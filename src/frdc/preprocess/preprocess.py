import numpy as np


def dummy_preprocess(ar: np.ndarray):
    """ A dummy preprocessing function. This simply takes an ar and copies it on another axis

    Args:
        ar: Image NDArray, must be of shape (H, W, C)

    Returns:
        A list of preprocessed NDArray, of shape (H, W, C) each.

    """
    return [ar, ar]
