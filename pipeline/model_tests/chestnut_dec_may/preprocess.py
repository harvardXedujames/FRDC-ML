import numpy as np
import torch
from glcm_cupy import Features
from torchvision.transforms.v2 import Resize

from frdc.models import FaceNet
from frdc.preprocess.glcm_padded import append_glcm_padded_cached
from frdc.preprocess.scale import scale_normal_per_band, scale_0_1_per_band


# TODO: Eventually, we will have multiple tests, and we should try to make
#   this function test agnostic.


def channel_preprocess(ar: np.ndarray) -> np.ndarray:
    # Preprocesses a channel array of shape: (H, W)
    shape = ar.shape
    ar_flt = ar.flatten()
    return ar_flt.reshape(*shape)


def segment_preprocess(ar: np.ndarray) -> torch.Tensor:
    # Preprocesses a segment array of shape: (H, W, C)

    # Add a small epsilon to avoid upper bound of 1.0
    ar = scale_0_1_per_band(ar, epsilon=0.001)
    ar = append_glcm_padded_cached(
        ar, step_size=7, bin_from=1, bin_to=128, radius=3, features=(Features.MEAN,)
    )
    # We can then scale normal for better neural network convergence
    ar = scale_normal_per_band(ar)

    # TODO: Doesn't seem like we have any channel preprocessing here.
    # ar = np.stack([
    #     channel_preprocess(ar[..., ch]) for ch in range(ar.shape[-1])
    # ])

    t = torch.from_numpy(ar)
    t = Resize([FaceNet.MIN_SIZE, FaceNet.MIN_SIZE], antialias=True)(t)
    return t


def preprocess(l_ar: list[np.ndarray]) -> torch.Tensor:
    """Preprocesses a list of segments.

    Notes:
        We structure the transformations into 3 levels.
        1. Segments transformation (This function)
        2. Per segment transformation (segment_preprocess)
        3. Per channel transformation (channel_preprocess)

    Returns:
        A preprocessed tensor of shape: (batch, channels, height, width)
    """

    l_t: list[torch.Tensor] = [segment_preprocess(ar) for ar in l_ar]
    t: torch.Tensor = torch.stack(l_t)
    t = torch.nan_to_num(t)
    return t
