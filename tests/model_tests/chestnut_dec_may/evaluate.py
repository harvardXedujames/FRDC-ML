import lightning as pl
import numpy as np
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip

from frdc.load import FRDCDataset
from model_tests.chestnut_dec_may.main import InceptionV3Module, preprocess

# Get our Test
# TODO: Ideally, we should have a separate dataset for testing.


def get_ds(h_flip, v_flip):
    return FRDCDataset(
        "chestnut_nature_park",
        "20210510",
        "90deg43m85pct255deg/map",
        transform=Compose(
            [
                preprocess,
                RandomHorizontalFlip(p=h_flip),
                RandomVerticalFlip(p=v_flip),
            ]
        ),
    )


def main():
    ds = ConcatDataset(
        [get_ds(0, 0), get_ds(1, 0), get_ds(0, 1), get_ds(1, 1)]
    )

    m = InceptionV3Module.load_from_checkpoint(
        "lightning_logs/version_20/checkpoints/epoch=10-step=1100.ckpt"
    )
    # Make predictions
    trainer = pl.Trainer()
    pred = trainer.predict(m, dataloaders=DataLoader(ds, batch_size=32))
    y_trues = []
    y_preds = []
    for y_true, y_pred in pred:
        y_trues.append(y_true)
        y_preds.append(y_pred.argmax(dim=1))
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    acc = (y_trues == y_preds).mean()

    # Plot the confusion matrix
    cm = confusion_matrix(y_trues, y_preds)

    plt.figure(figsize=(10, 10))

    heatmap(
        cm,
        annot=True,
        xticklabels=m.oe.categories_[0],
        yticklabels=m.oe.categories_[0],
        cbar=False,
    )
    plt.title(f"Accuracy: {acc:.2%}")

    plt.tight_layout()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
