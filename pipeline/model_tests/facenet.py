""" Tests for the FaceNet model.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import TensorDataset, Dataset, Subset
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomVerticalFlip, \
    Resize

from frdc.models import FaceNet
from frdc.train import FRDCDataModule, FRDCModule
from pipeline.model_tests.utils import Functional as F
from pipeline.model_tests.utils import get_dataset


# See FRDCDataModule for fn_segment_tf and fn_split
def preprocess(l_ar: list[np.ndarray]) -> torch.Tensor:
    """ We structure the transformations into 3 levels.
    1. Segments transformation
    2. Per segment transformation
    3. Per channel transformation
    """

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
        ar = F.scale_0_1_per_band(ar) / 1.001
        ar = F.glcm(ar)
        ar = np.stack([
            channel_preprocess(ar[..., ch]) for ch in range(ar.shape[-1])
        ])

        t = torch.from_numpy(ar)
        t = Resize([FaceNet.MIN_SIZE, FaceNet.MIN_SIZE],
                   antialias=True)(t)
        return t

    l_t: list[torch.Tensor] = [segment_preprocess(ar) for ar in l_ar]
    t: torch.Tensor = torch.stack(l_t)
    t = torch.nan_to_num(t)
    return t


def augmentation(t: torch.Tensor) -> torch.Tensor:
    t = RandomHorizontalFlip()(t)
    t = RandomVerticalFlip()(t)
    return t


def train_val_test_split(x: TensorDataset) -> list[Dataset, Dataset, Dataset]:
    return [
        Subset(x, list(range(len(segments_0)))),
        Subset(x, list(
            range(len(segments_0), len(segments_0) + len(segments_1)))),
        Subset(x, [])
    ]


segments_0, labels_0 = get_dataset(
    'chestnut_nature_park', '20201218', None
)
segments_1, labels_1 = get_dataset(
    'chestnut_nature_park', '20210510', '90deg43m85pct255deg/map'
)
segments = [*segments_0, *segments_1]
labels = [*labels_0, *labels_1]
BATCH_SIZE = 5
EPOCHS = 100
LR = 1e-3

dm = FRDCDataModule(
    segments=segments,
    labels=labels,
    preprocess=preprocess,
    augmentation=augmentation,
    train_val_test_split=train_val_test_split,
    batch_size=BATCH_SIZE
)

trainer = pl.Trainer(
    max_epochs=EPOCHS, deterministic=True,
    log_every_n_steps=4,
    callbacks=[
        LearningRateMonitor(logging_interval='epoch')
    ]
)

# TODO: A bit hacky, but if we defined a LOAD_FROM, we load from that
#   checkpoint. Otherwise, we train from scratch.
LOAD_FROM = None  # "lightning_logs/version_30/checkpoints/epoch=99-step=700.ckpt"

if LOAD_FROM is not None:
    m = FRDCModule.load_from_checkpoint(checkpoint_path=LOAD_FROM)
    dm.setup('validate')
else:
    m = FRDCModule(
        model_cls=FaceNet,
        model_kwargs=dict(n_out_classes=len(set(labels))),
        optim_cls=torch.optim.Adam,
        optim_kwargs=dict(
            lr=LR,
            weight_decay=1e-4,
            amsgrad=True,
        )
    )
    trainer.fit(m, datamodule=dm)
# %%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from seaborn import heatmap

preds = trainer.predict(m, dataloaders=dm.val_dataloader())
preds = torch.concat(preds, dim=0)

cm = confusion_matrix(dm.le.transform(labels_1), preds.argmax(dim=1))
plt.figure(figsize=(10, 10))

heatmap(
    cm, annot=True,
    xticklabels=dm.le.classes_,
    yticklabels=dm.le.classes_,
    cbar=False)

plt.tight_layout(pad=3)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix2.png')
