""" Tests for the InceptionV3 model on the Chestnut Nature Park dataset.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""
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

from frdc.load.preset import FRDCDatasetPreset as ds
from frdc.models.inceptionv3 import InceptionV3MixMatchModule
from frdc.train.frdc_datamodule import FRDCDataModule
from frdc.utils.training import predict, plot_confusion_matrix
from model_tests.utils import (
    train_preprocess_augment,
    train_unl_preprocess,
    val_preprocess,
    FRDCDatasetFlipped,
)


# Uncomment this to run the W&B monitoring locally
# import os
#
# os.environ["WANDB_MODE"] = "offline"


def get_y_encoder(targets):
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    oe.fit(np.array(targets).reshape(-1, 1))
    return oe


def get_x_scaler(segments):
    ss = StandardScaler()
    ss.fit(
        np.concatenate([segm.reshape(-1, segm.shape[-1]) for segm in segments])
    )
    return ss


def main(
    batch_size=32,
    epochs=10,
    train_iters=25,
    lr=1e-3,
    wandb_name="chestnut_dec_may",
    wandb_project="frdc",
):
    # Prepare the dataset
    train_lab_ds = ds.chestnut_20201218(transform=train_preprocess_augment)
    train_unl_ds = ds.chestnut_20201218.unlabelled(
        transform=train_unl_preprocess(2)
    )
    val_ds = ds.chestnut_20210510_43m(transform=val_preprocess)

    # Prepare the datamodule and trainer
    dm = FRDCDataModule(
        train_lab_ds=train_lab_ds,
        train_unl_ds=train_unl_ds,  # None to use supervised DM
        val_ds=val_ds,
        batch_size=batch_size,
        train_iters=train_iters,
        sampling_strategy="random",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        deterministic=True,
        accelerator="gpu",
        log_every_n_steps=4,
        callbacks=[
            # Stop training if the validation loss doesn't improve for 4 epochs
            EarlyStopping(monitor="val/ce_loss", patience=4, mode="min"),
            # Log the learning rate on TensorBoard
            LearningRateMonitor(logging_interval="epoch"),
            # Save the best model
            ckpt := ModelCheckpoint(
                monitor="val/ce_loss", mode="min", save_top_k=1
            ),
        ],
        logger=(
            logger := WandbLogger(
                name=wandb_name,
                project=wandb_project,
            )
        ),
    )

    oe = get_y_encoder(train_lab_ds.targets)
    ss = get_x_scaler(train_lab_ds.ar_segments)

    m = InceptionV3MixMatchModule(
        in_channels=train_lab_ds.ar.shape[-1],
        n_classes=len(oe.categories_[0]),
        lr=lr,
        x_scaler=ss,
        y_encoder=oe,
        imagenet_scaling=True,
    )

    trainer.fit(m, datamodule=dm)

    with open(Path(__file__).parent / "report.md", "w") as f:
        f.write(
            f"# Chestnut Nature Park (Dec 2020 vs May 2021)\n"
            f"- Results: [WandB Report]({wandb.run.get_url()})"
        )

    y_true, y_pred = predict(
        ds=FRDCDatasetFlipped(
            "chestnut_nature_park",
            "20210510",
            "90deg43m85pct255deg",
            transform=val_preprocess,
        ),
        model_cls=InceptionV3MixMatchModule,
        ckpt_pth=Path(ckpt.best_model_path),
    )
    fig, ax = plot_confusion_matrix(y_true, y_pred, oe.categories_[0])
    acc = np.sum(y_true == y_pred) / len(y_true)
    ax.set_title(f"Accuracy: {acc:.2%}")

    wandb.log({"eval/confusion_matrix": wandb.Image(fig)})
    wandb.log({"eval/eval_accuracy": acc})

    wandb.finish()


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 50
    TRAIN_ITERS = 25
    LR = 1e-3

    torch.set_float32_matmul_precision("high")
    main(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        train_iters=TRAIN_ITERS,
        lr=LR,
        wandb_name="Try with Inception Unfrozen & Random Erasing",
        wandb_project="frdc-dev",
    )
