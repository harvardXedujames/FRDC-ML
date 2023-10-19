import numpy as np
import torch
from torchvision.transforms.v2 import Resize

from frdc.models import FaceNet
from frdc.preprocess import scale_0_1_per_band
from pipeline.model_tests.chestnut_dec_may.utils import Functional as F


def channel_preprocess(ar: np.ndarray) -> np.ndarray:
    # Preprocesses a channel array of shape: (H, W)
    shape = ar.shape
    ar_flt = ar.flatten()
    ar_flt = F.whiten(ar_flt)
    return ar_flt.reshape(*shape)


def segment_preprocess(ar: np.ndarray) -> torch.Tensor:
    # Preprocesses a segment array of shape: (H, W, C)

    # We divide by 1.001 is make the range [0, 1) instead of [0, 1] so that
    # glcm_padded can work properly.
    ar = scale_0_1_per_band(ar) / 1.001
    ar = F.glcm(ar)
    ar = np.stack([
        channel_preprocess(ar[..., ch]) for ch in range(ar.shape[-1])
    ])

    t = torch.from_numpy(ar)
    t = Resize([FaceNet.MIN_SIZE, FaceNet.MIN_SIZE],
               antialias=True)(t)
    return t


def preprocess(l_ar: list[np.ndarray]) -> torch.Tensor:
    """ Preprocesses a list of segments.

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
