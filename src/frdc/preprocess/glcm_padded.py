from typing import Hashable

import numpy as np
from glcm_cupy import glcm

from frdc.conf import ROOT_DIR
from frdc.utils.file_cache import file_cache


def make_hashable(x: object) -> Hashable:
    return x if isinstance(x, np.ndarray) else str(x)


@file_cache(fn_cache_fp=lambda x: ROOT_DIR / ".cache" / f"glcm_{x}.npy",
            fn_load_object=np.load,
            fn_save_object=np.save,
            fn_make_hashable=make_hashable)
def glcm_padded(
        ar: np.ndarray,
        *,
        bin_from: int,
        bin_to: int,
        radius: int,
        step_size: int = 1,
        **kwargs
) -> np.ndarray:
    """ A wrapper for glcm-cupy's glcm. This pads the GLCM automatically

    Notes:
        This function is also cached. The cache is invalidated if any of the
        arguments change.

    Args:
        ar: Array to compute GLCM on. Must be of shape (H, W, C)
        bin_from: The upper bounds integer for the input
        bin_to: The resolution of the GLCM
        radius: Radius of each GLCM window
        step_size: Step size of each GLCM window
        **kwargs: Additional arguments to pass to glcm-cupy's glcm

    Returns:
        Given an input (H, W, C), returns a
        GLCM of shape (H, W, C, GLCM Features)
    """
    pad = radius + step_size
    ar_pad = np.pad(
        ar,
        pad_width=((pad,), (pad,), (0,)),
        constant_values=np.nan
    )
    return glcm(
        ar_pad,
        bin_from=bin_from,
        bin_to=bin_to,
        radius=radius,
        step_size=step_size,
        **kwargs
    )
