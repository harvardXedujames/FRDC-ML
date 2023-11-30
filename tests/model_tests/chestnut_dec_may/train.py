""" Tests for the InceptionV3 model on the Chestnut Nature Park dataset.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""
import os
from pathlib import Path
from typing import Any

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
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    RandomHorizontalFlip,
    RandomCrop,
    CenterCrop,
)

from frdc.load import FRDCDataset
from frdc.models import InceptionV3
from frdc.train.frdc_datamodule_new import FRDCDataModule


def preprocess(x):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            CenterCrop(
                [InceptionV3.MIN_SIZE, InceptionV3.MIN_SIZE],
            ),
        ]
    )(x)


def random_preprocess(x):
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
        ]
    )(x)


class InceptionV3Module(InceptionV3):
    def __init__(
        self,
        *,
        n_out_classes: int,
        lr: float,
        ss: StandardScaler,
        oe: OrdinalEncoder,
    ):
        self.lr = lr
        self.ss = ss
        self.oe = oe
        super().__init__(n_out_classes=n_out_classes)

    # TODO: PyTorch Lightning is complaining that I'm setting this in the
    #       Module instead of DataModule, we can likely migrate this to
    #       the ___step() functions.
    @torch.no_grad()
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        x, y = batch

        # Standard Scaler only accepts (n_samples, n_features), so we need to
        # do some fancy reshaping.
        # Note that moving dimensions then reshaping is different from just
        # reshaping!
        x: torch.Tensor
        b, c, h, w = x.shape

        # Move Channel to the last dimension then transform
        x_ss: np.ndarray = self.ss.transform(
            x.permute(0, 2, 3, 1).reshape(-1, c)
        )

        # Move Channel back to the second dimension
        x_: torch.Tensor = (
            torch.from_numpy(x_ss.reshape(b, h, w, c))
            .permute(0, 3, 1, 2)
            .float()
        )

        # Ordinal Encoder only accepts (n_samples, 1), so we need to do some
        y: tuple[str]
        y_: torch.Tensor = torch.from_numpy(
            self.oe.transform(np.array(y).reshape(-1, 1)).squeeze()
        )

        # Ordinal Encoders can return a np.nan if the value is not in the
        # categories. We will remove that from the batch.
        x_ = x_[~torch.isnan(y_)]
        y_ = y_[~torch.isnan(y_)]

        return x_, y_.long()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim


def main():
    run = wandb.init()
    logger = WandbLogger(name="chestnut_dec_may", project="frdc")
    # Prepare the dataset
    train_ds = FRDCDataset(
        "chestnut_nature_park",
        "20201218",
        None,
        transform=random_preprocess,
    )
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
    oe.fit(np.array(train_ds.targets).reshape(-1, 1))
    n_classes = len(oe.categories_[0])

    ss = StandardScaler()
    ss.fit(train_ds.ar.reshape(-1, train_ds.ar.shape[-1]))

    # Prepare the datamodule and trainer
    dm = FRDCDataModule(
        train_ds=train_ds,
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
    m = InceptionV3Module(
        n_out_classes=n_classes,
        lr=LR,
        ss=ss,
        oe=oe,
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
    TRAIN_ITERS = 100
    VAL_ITERS = 10
    LR = 1e-3
    os.environ["GOOGLE_CLOUD_PROJECT"] = "frmodel"
    assert wandb.run is None
    #
    wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))
    main()
