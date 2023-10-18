""" Tests for the FaceNet model.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset. The test accuracy is then reported into the logs.
"""

import lightning as pl
import numpy as np
import torch
import torchvision
from glcm_cupy import Features
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, Dataset, Subset
from torchvision.transforms.functional import resized_crop

from frdc.load import FRDCDataset
from frdc.models import FaceNet
from frdc.preprocess import (
    extract_segments_from_bounds, scale_static_per_band, scale_0_1_per_band
)
from frdc.preprocess.glcm_padded import glcm_padded_cached
from frdc.train import FRDCDataModule, FRDCModule

BANDS = ['NB', 'NG', 'NR', 'RE', 'NIR']


class F:
    @staticmethod
    def scale_static_per_band(ar: np.ndarray) -> np.ndarray:
        return scale_static_per_band(ar, BANDS)

    @staticmethod
    def scale_0_1_per_band(ar: np.ndarray) -> np.ndarray:
        return scale_0_1_per_band(ar)

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
    def pca_threshold(x: np.ndarray,
                      n_components: int = 5) -> np.ndarray:
        """ PCA thresholding

        Args:
            x: The array to threshold
            n_components: The number of components to keep

        Returns:
            The thresholded array
        """

        pca = PCA(n_components=n_components)
        pca_x = pca.fit_transform(x)
        return pca.inverse_transform(pca_x)

    @staticmethod
    def center_crop(t: torch.Tensor,
                    size: int = FaceNet.MIN_SIZE) -> torch.Tensor:
        return torchvision.transforms.functional.center_crop(t, size)

    @staticmethod
    def random_resized_crop(ar: np.ndarray,
                            size: int = FaceNet.MIN_SIZE) -> torch.Tensor:
        rrc = torchvision.transforms.RandomResizedCrop(
            size=size,
            # scale=((64 / 250) ** 2, 1),
            # ratio=(1 / 2, 2),
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
    def random_rotation(t: torch.Tensor,
                        degrees: float = 10) -> torch.Tensor:
        rr = torchvision.transforms.RandomRotation(
            degrees=degrees,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            expand=True
        )

        return rr(t)

    @staticmethod
    def random_flip(t: torch.Tensor) -> torch.Tensor:
        rhf = torchvision.transforms.RandomHorizontalFlip()
        rvf = torchvision.transforms.RandomVerticalFlip()
        return rhf(rvf(t))

    @staticmethod
    def l2_norm(ar: np.ndarray) -> np.ndarray:
        return ar / np.linalg.norm(ar, ord=2)

    @staticmethod
    def whiten(ar: np.ndarray) -> np.ndarray:
        return (ar - np.mean(ar)) / np.std(ar)


# See FRDCDataModule for fn_segment_tf and fn_split
def fn_segments_tf(l_ar: list[np.ndarray]) -> torch.Tensor:
    """ We structure the transformations into 3 levels.
    1. Segments transformation
    2. Per segment transformation
    3. Per channel transformation
    """

    def chn_tf(ar: np.ndarray) -> np.ndarray:
        shape = ar.shape
        ar_flt = ar.flatten()
        ar_flt = F.l2_norm(ar_flt)
        return ar_flt.reshape(*shape)

    def segment_tf(ar: np.ndarray) -> torch.Tensor:
        ar = F.scale_static_per_band(ar)
        ar = F.glcm(ar)
        ar = np.stack([chn_tf(ar[..., ch]) for ch in range(ar.shape[-1])])
        t = torch.from_numpy(ar)
        t = F.random_rotation(t)
        t = F.random_crop(t)
        t = F.random_flip(t)

        return t

    l_t: list[torch.Tensor] = [segment_tf(ar) for ar in l_ar]
    t: torch.Tensor = torch.stack(l_t)
    t = torch.nan_to_num(t)
    return t


BATCH_SIZE = 5
EPOCHS = 100


def get_dataset(site, date, version):
    ds = FRDCDataset(site=site, date=date, version=version)
    ar, order = ds.get_ar_bands(BANDS)
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar, bounds, cropped=True)
    return segments, labels


segments_0, labels_0 = get_dataset(
    'chestnut_nature_park', '20201218', None
)
segments_1, labels_1 = get_dataset(
    'chestnut_nature_park', '20210510', '90deg43m85pct255deg/map'
)
segments = [*segments_0, *segments_1]
labels = [*labels_0, *labels_1]


def fn_split(x: TensorDataset) -> list[Dataset, Dataset, Dataset]:
    # random_split()
    return [
        Subset(x, list(range(len(segments_0)))),
        Subset(x, list(
            range(len(segments_0), len(segments_0) + len(segments_1)))),
        Subset(x, [])
    ]
    # x,
    # lengths=[len(segments_0), len(segments_1), 0]
    # )


# %%
dm = FRDCDataModule(
    segments=segments,
    labels=labels,
    fn_segment_tf=fn_segments_tf,
    fn_split=fn_split,
    batch_size=BATCH_SIZE
)
# %%
m = FRDCModule(
    model=FaceNet(
        n_in_channels=8,
        n_out_classes=len(set(labels))
    ),
    optimizer=lambda x: torch.optim.Adam(
        x.parameters(),
        lr=1e-3,
        weight_decay=5e-4
    )
)

trainer = pl.Trainer(max_epochs=EPOCHS)
trainer.fit(m, datamodule=dm)