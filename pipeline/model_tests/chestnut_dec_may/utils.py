import numpy as np
import torch
import torchvision
from glcm_cupy import Features
from torchvision.transforms.functional import resized_crop

from frdc.load import FRDCDataset
from frdc.models import FaceNet
from frdc.preprocess import (
    scale_static_per_band, scale_0_1_per_band, extract_segments_from_bounds
)
from frdc.preprocess.glcm_padded import glcm_padded_cached

BANDS = ['NB', 'NG', 'NR', 'RE', 'NIR']


def get_dataset(site, date, version):
    ds = FRDCDataset(site=site, date=date, version=version)
    ar, order = ds.get_ar_bands(BANDS)
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar, bounds, cropped=True)
    return segments, labels


class Functional:


    @staticmethod
    def glcm(ar: np.ndarray,
             step_size=7,
             bin_from=1,
             bin_to=128,
             radius=3,
             features=(Features.MEAN,)) -> np.ndarray:
        ar_glcm = glcm_padded_cached(
            ar,
            step_size=step_size, bin_from=bin_from, bin_to=bin_to,
            radius=radius, features=features
        )

        # : (H, W, C), s_glcm: (H, W, C, GLCM Features)
        # Concatenate the GLCM features to the channels
        ar = np.concatenate([ar[..., np.newaxis], ar_glcm], axis=-1)
        return ar.reshape(*ar.shape[:2], -1)

    @staticmethod
    def center_crop(t: torch.Tensor,
                    size: int = FaceNet.MIN_SIZE) -> torch.Tensor:
        return torchvision.transforms.functional.center_crop(t, size)

    @staticmethod
    def random_resized_crop(ar: np.ndarray,
                            size: int = FaceNet.MIN_SIZE) -> torch.Tensor:
        rrc = torchvision.transforms.RandomResizedCrop(
            size=size,
            scale=((64 / 250) ** 2, 1),
            ratio=(1 / 2, 2),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True
        )

        return rrc(ar)

    @staticmethod
    def random_crop(t: torch.Tensor,
                    size: int = FaceNet.MIN_SIZE) -> torch.Tensor:
        rc = torchvision.transforms.RandomCrop(
            size=size,
            pad_if_needed=True,
        )

        return rc(t)

    @staticmethod
    def whiten(ar: np.ndarray) -> np.ndarray:
        return (ar - np.nanmean(ar)) / np.nanstd(ar)
