""" Tests for the FaceNet model.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from torch.utils.data import TensorDataset, Dataset, Subset

from frdc.models import InceptionV3
from frdc.train import FRDCDataModule, FRDCModule
from tests.model_tests.chestnut_dec_may.augmentation import augmentation
from tests.model_tests.chestnut_dec_may.preprocess import preprocess
from tests.model_tests.utils import get_dataset
from lightning.pytorch.loggers import WandbLogger
import wandb

assert wandb.run is None

wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))
run = wandb.init()
logger = WandbLogger(name="chestnut_dec_may", project="frdc")


def train_val_test_split(
    x: TensorDataset,
) -> list[Dataset, Dataset, Dataset]:
    # Defines how to split the dataset into train, val, test subsets.
    # TODO: Quite ugly as it uses the global variables segments_0 and
    #  segments_1. Will need to refactor this.
    return [
        Subset(x, list(range(len(segments_0)))),
        Subset(
            x,
            list(range(len(segments_0), len(segments_0) + len(segments_1))),
        ),
        [],
    ]


# Prepare the dataset
segments_0, labels_0 = get_dataset("chestnut_nature_park", "20201218", None)
segments_1, labels_1 = get_dataset(
    "chestnut_nature_park", "20210510", "90deg43m85pct255deg/map"
)


# Concatenate the datasets
segments = [*segments_0, *segments_1]
labels = [*labels_0, *labels_1]

BATCH_SIZE = 5
EPOCHS = 50
LR = 1e-3

# Prepare the datamodule and trainer
dm = FRDCDataModule(
    # Input to the model
    segments=segments,
    # Output of the model
    labels=labels,
    # Preprocessing function
    preprocess=preprocess,
    # Augmentation function (Only on train)
    augmentation=augmentation,
    # Splitting function
    train_val_test_split=train_val_test_split,
    # Batch size
    batch_size=BATCH_SIZE,
)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    # fast_dev_run=True,
    # Set the seed for reproducibility
    # TODO: Though this is set, the results are still not reproducible.
    deterministic=True,
    # fast_dev_run=True,
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
    # Our model is the "FaceNet" model
    # TODO: It's not really the FaceNet model,
    #  but a modified version of it.
    model_cls=InceptionV3,
    model_kwargs=dict(n_out_classes=len(set(labels))),
    # We use the Adam optimizer
    optim_cls=torch.optim.Adam,
    # TODO: This is not fine-tuned.
    optim_kwargs=dict(lr=LR, weight_decay=1e-4, amsgrad=True),
)

trainer.fit(m, datamodule=dm)
# TODO: Quite hacky, but we need to save the label encoder for prediction.
np.save("le.npy", dm.le.classes_)

report = f"""
# Chestnut Nature Park (Dec 2020 vs May 2021)
[WandB Report]({run.get_url()})
TODO: Authentication for researchers
"""

with open(Path(__file__).parent / "report.md", "w") as f:
    f.write(report)


wandb.finish()
