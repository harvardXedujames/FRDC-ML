""" Tests for the FaceNet model.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""
import os
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    RandomHorizontalFlip,
    Normalize,
    RandomCrop,
)

from frdc.load import FRDCDataset
from frdc.load.dataset import FRDCConcatDataset
from frdc.models import InceptionV3
from frdc.train import FRDCModule
from frdc.train.frdc_datamodule_new import FRDCDataModule, Transforms

os.environ["GOOGLE_CLOUD_PROJECT"] = "frmodel"
assert wandb.run is None

wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))
run = wandb.init()
logger = WandbLogger(name="chestnut_dec_may", project="frdc")

# Prepare the dataset
ds0 = FRDCDataset(
    "chestnut_nature_park",
    "20201218",
    None,
)
ds1 = FRDCDataset(
    "chestnut_nature_park", "20210510", "90deg43m85pct255deg/map"
)

ds = FRDCConcatDataset([ds0, ds1])
n_classes = len(set(ds.targets))

BATCH_SIZE = 32
EPOCHS = 50

# We only use the mean and std of the train set. This is to simulate blind
# testing.
mean = np.nanmean(ds0.ar, axis=(0, 1))
std = np.nanstd(ds0.ar, axis=(0, 1))


def tf(x):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RandomCrop(
                [InceptionV3.MIN_SIZE, InceptionV3.MIN_SIZE],
                pad_if_needed=True,
                padding_mode="constant",
                fill=0,
            ),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            Normalize(mean=mean, std=std),
        ]
    )(x)


# Prepare the datamodule and trainer
dm = FRDCDataModule(
    ds=ds,
    transforms=Transforms(
        train_tf=tf,
        val_tf=tf,
    ),
    batch_size=BATCH_SIZE,
    train_iters=25,
    val_iters=25,
)

LR = 1e-3
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    # Set the seed for reproducibility
    # TODO: Though this is set, the results are still not reproducible.
    deterministic=True,
    accelerator="gpu",
    log_every_n_steps=4,
    callbacks=[
        # Stop training if the validation loss doesn't improve for 4 epochs
        EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        # Log the learning rate on TensorBoard
        LearningRateMonitor(logging_interval="epoch"),
        # Save the best model
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
    ],
    logger=logger,
)
m = FRDCModule(
    model_f=lambda: InceptionV3(n_out_classes=n_classes),
    optim_f=lambda model: torch.optim.Adam(model.parameters(), lr=LR),
    scheduler_f=lambda optim: torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optim,
        gamma=1,
    ),
    le=dm.le,
)

trainer.fit(m, datamodule=dm)

report = f"""
# Chestnut Nature Park (Dec 2020 vs May 2021)
[WandB Report]({run.get_url()})
TODO: Authentication for researchers
"""

with open(Path(__file__).parent / "report.md", "w") as f:
    f.write(report)

wandb.finish()
