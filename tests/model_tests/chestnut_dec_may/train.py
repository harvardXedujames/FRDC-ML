""" Tests for the InceptionV3 model on the Chestnut Nature Park dataset.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""

# Uncomment this to run the W&B monitoring locally
# import os
# os.environ["WANDB_MODE"] = "offline"

from pathlib import Path

import lightning as pl
import numpy as np
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from frdc.load.dataset import FRDCUnlabelledDataset, FRDCDatasetPreset
from frdc.models.inceptionv3 import InceptionV3MixMatchModule
from frdc.train.frdc_datamodule import FRDCDataModule
from model_tests.utils import (
    train_preprocess,
    train_unl_preprocess,
    preprocess,
    evaluate,
    FRDCDatasetFlipped,
)


def main(
    batch_size=32,
    epochs=10,
    train_iters=25,
    val_iters=15,
    lr=1e-3,
):
    run = wandb.init()
    logger = WandbLogger(name="chestnut_dec_may", project="frdc")
    # Prepare the dataset
    train_lab_ds = FRDCDatasetPreset.chestnut_20201218(
        transform=train_preprocess
    )

    # TODO: This is a hacky impl of the unlabelled dataset, see the docstring
    #       for future work.
    train_unl_ds = FRDCUnlabelledDataset(
        "chestnut_nature_park",
        "20201218",
        None,
        transform=train_unl_preprocess(2),
    )

    val_ds = FRDCDatasetPreset.chestnut_20210510_43m(transform=preprocess)

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
        batch_size=batch_size,
        train_iters=train_iters,
        val_iters=val_iters,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        deterministic=True,
        accelerator="gpu",
        log_every_n_steps=4,
        callbacks=[
            # Stop training if the validation loss doesn't improve for 4 epochs
            EarlyStopping(monitor="val_loss", patience=4, mode="min"),
            # Log the learning rate on TensorBoard
            LearningRateMonitor(logging_interval="epoch"),
            # Save the best model
            ckpt := ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1
            ),
        ],
        logger=logger,
    )
    m = InceptionV3MixMatchModule(
        n_classes=n_classes,
        lr=lr,
        x_scaler=ss,
        y_encoder=oe,
    )

    trainer.fit(m, datamodule=dm)

    with open(Path(__file__).parent / "report.md", "w") as f:
        f.write(
            f"# Chestnut Nature Park (Dec 2020 vs May 2021)\n"
            f"- Results: [WandB Report]({run.get_url()})"
        )

    fig, acc = evaluate(
        ds=FRDCDatasetFlipped(
            "chestnut_nature_park",
            "20210510",
            "90deg43m85pct255deg",
            transform=preprocess,
        ),
        ckpt_pth=Path(ckpt.best_model_path),
    )
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    wandb.log({"eval_accuracy": acc})

    wandb.finish()


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 10
    TRAIN_ITERS = 25
    VAL_ITERS = 15
    LR = 1e-3

    assert wandb.run is None
    wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))
    main(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        train_iters=TRAIN_ITERS,
        val_iters=VAL_ITERS,
        lr=LR,
    )
