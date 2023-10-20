import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

from frdc.train import FRDCDataModule
from frdc.train import FRDCModule
from pipeline.model_tests.chestnut_dec_may.preprocess import preprocess
from pipeline.model_tests.chestnut_dec_may.utils import get_dataset

# Get our Test
# TODO: Ideally, we should have a separate dataset for testing.
segments, labels = get_dataset(
    "chestnut_nature_park", "20210510", "90deg43m85pct255deg/map"
)

# Prepare the datamodule and trainer
dm = FRDCDataModule(segments=segments, preprocess=preprocess, batch_size=5)

# TODO: Hacky way to load our LabelEncoder
dm.le.classes_ = np.load("le.npy", allow_pickle=True)

# Load the model
m = FRDCModule.load_from_checkpoint(
    "lightning_logs/version_88/checkpoints/epoch=99-step=700.ckpt"
)

# Make predictions
trainer = pl.Trainer(logger=False)
pred = trainer.predict(m, datamodule=dm)
y_pred = torch.concat(pred, dim=0).argmax(dim=1)
y_true = dm.le.transform(labels)

# Plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 10))

heatmap(
    cm,
    annot=True,
    xticklabels=dm.le.classes_,
    yticklabels=dm.le.classes_,
    cbar=False,
)

plt.tight_layout(pad=3)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
