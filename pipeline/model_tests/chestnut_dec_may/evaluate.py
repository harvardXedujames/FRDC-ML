import lightning as pl
import matplotlib.pyplot as plt
import torch
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

from frdc.train import FRDCDataModule
from frdc.train import FRDCModule
from pipeline.model_tests.chestnut_dec_may.augmentation import augmentation
from pipeline.model_tests.chestnut_dec_may.preprocess import preprocess
from pipeline.model_tests.chestnut_dec_may.utils import get_dataset

segments, labels = get_dataset(
    'chestnut_nature_park', '20210510', '90deg43m85pct255deg/map'
)
dm = FRDCDataModule(
    segments=segments,
    labels=labels,
    preprocess=preprocess,
    augmentation=augmentation,
    train_val_test_split=lambda x: [[], x, []],
    batch_size=5
)

trainer = pl.Trainer(logger=False)
m = FRDCModule.load_from_checkpoint(
    "lightning_logs/version_88/checkpoints/epoch=99-step=700.ckpt"
)

dm.setup('validate')

pred = trainer.predict(m, dataloaders=dm.val_dataloader())
y_pred = torch.concat(pred, dim=0).argmax(dim=1)
y_true = dm.le.transform(labels)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 10))

heatmap(
    cm, annot=True,
    xticklabels=dm.le.classes_,
    yticklabels=dm.le.classes_,
    cbar=False)

plt.tight_layout(pad=3)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix3.png')
