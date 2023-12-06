""" Tests for the InceptionV3 model on the Chestnut Nature Park dataset.

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
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomCrop,
    CenterCrop,
)

from frdc.load import FRDCDataset
from frdc.load.dataset import FRDCUnlabelledDataset
from frdc.models.inceptionv3 import (
    InceptionV3MixMatchModule,
)
from frdc.train.frdc_datamodule import FRDCDataModule


def preprocess(x):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            CenterCrop(
                [
                    InceptionV3MixMatchModule.MIN_SIZE,
                    InceptionV3MixMatchModule.MIN_SIZE,
                ],
            ),
        ]
    )(x)


def train_preprocess(x):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RandomCrop(
                [
                    InceptionV3MixMatchModule.MIN_SIZE,
                    InceptionV3MixMatchModule.MIN_SIZE,
                ],
                pad_if_needed=True,
                padding_mode="constant",
                fill=0,
            ),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ]
    )(x)


def train_unl_preprocess(x):
    # This simulates the n_aug of MixMatch
    return train_preprocess(x), train_preprocess(x)


def main():
    run = wandb.init()
    logger = WandbLogger(name="chestnut_dec_may", project="frdc")
    # Prepare the dataset
    train_lab_ds = FRDCDataset(
        "chestnut_nature_park",
        "20201218",
        None,
        transform=train_preprocess,
    )

    # TODO: This is a hacky impl of the unlabelled dataset, see the docstring
    #       for future work.
    train_unl_ds = FRDCUnlabelledDataset(
        "chestnut_nature_park",
        "20201218",
        None,
        transform=train_unl_preprocess,
    )

    # Subset(train_ds, np.argwhere(train_ds.targets == 0).reshape(-1))
    val_ds = FRDCDataset(
        "chestnut_nature_park",
        "20210510",
        "90deg43m85pct255deg/map",
        transform=preprocess,
    )

    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    oe.fit(np.array(train_lab_ds.targets).reshape(-1, 1))
    n_classes = len(oe.categories_[0])

    ss = StandardScaler()
    ss.fit(train_lab_ds.ar.reshape(-1, train_lab_ds.ar.shape[-1]))

    # Prepare the datamodule and trainer
    dm = FRDCDataModule(
        train_lab_ds=train_lab_ds,
        # Pass in None to use the default supervised DM
        train_unl_ds=train_unl_ds,
        val_ds=val_ds,
        batch_size=BATCH_SIZE,
        train_iters=TRAIN_ITERS,
        val_iters=VAL_ITERS,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
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
    m = InceptionV3MixMatchModule(
        n_classes=n_classes,
        lr=LR,
        x_scaler=ss,
        y_encoder=oe,
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


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 30
    TRAIN_ITERS = 50
    VAL_ITERS = 15
    LR = 1e-3
    os.environ["GOOGLE_CLOUD_PROJECT"] = "frmodel"

    assert wandb.run is None
    wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))
    main()
