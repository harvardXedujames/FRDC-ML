import logging

import lightning as pl
import torch
from skimage.transform import resize
from torch.utils.data import random_split
from torchvision.transforms.v2 import (
    Resize,
    Compose,
    ToImage,
    ToDtype,
)

from frdc.models import InceptionV3
from frdc.train import FRDCModule
from frdc.train.frdc_datamodule_new import FRDCDataModule, Transforms


def fn_segment_tf(x):
    x = [resize(s, [InceptionV3.MIN_SIZE, InceptionV3.MIN_SIZE]) for s in x]
    x = [torch.from_numpy(s) for s in x]
    x = torch.stack(x)
    x = torch.nan_to_num(x)
    x = x.permute(0, 3, 1, 2)
    return x


fn_split = lambda x: random_split(x, lengths=[len(x) - 6, 3, 3])

BATCH_SIZE = 3


def test_manual_segmentation_pipeline(ds) -> tuple[FRDCModule, FRDCDataModule]:
    """Manually segment the image according to bounds.csv,
    then train a model on it."""
    train_tf = Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize([InceptionV3.MIN_SIZE, InceptionV3.MIN_SIZE]),
        ]
    )

    dm = FRDCDataModule(
        ds=ds,
        transforms=Transforms(
            train_tf=lambda x: train_tf(x),
            val_tf=lambda x: train_tf(x),
            test_tf=lambda x: x,
        ),
        batch_size=BATCH_SIZE,
    )
    m = FRDCModule(
        model_f=lambda: InceptionV3(n_out_classes=10),
        optim_f=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        scheduler_f=lambda optim: torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optim,
            gamma=0.99,
        ),
    )

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(m, datamodule=dm)

    val_loss = trainer.validate(m, datamodule=dm)[0]["val_loss"]
    logging.debug(f"Validation score: {val_loss:.2%}")

    return m, dm
