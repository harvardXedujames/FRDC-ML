""" Tests for the FaceNet model.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""

import lightning as pl
import torch
from keras.src.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import TensorDataset, Dataset, Subset

from frdc.models import FaceNet
from frdc.train import FRDCDataModule, FRDCModule
from pipeline.model_tests.chestnut_dec_may.augmentation import augmentation
from pipeline.model_tests.chestnut_dec_may.preprocess import preprocess
from pipeline.model_tests.chestnut_dec_may.utils import get_dataset


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
        EarlyStopping(monitor='val_loss', patience=4, mode='min'),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
    ]
)
m = FRDCModule(
    model_cls=FaceNet,
    model_kwargs=dict(n_out_classes=len(set(labels))),
    optim_cls=torch.optim.Adam,
    optim_kwargs=dict(lr=LR, weight_decay=1e-4, amsgrad=True)
)
trainer.fit(m, datamodule=dm)
