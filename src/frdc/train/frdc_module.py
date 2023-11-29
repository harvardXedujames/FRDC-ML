from typing import Callable

import torch
from lightning import LightningModule
from sklearn.preprocessing import LabelEncoder
from torch import nn


class FRDCModule(LightningModule):
    def __init__(
        self,
        *,
        model: nn.Module,
        optim_f: Callable[[nn.Module], torch.optim.Optimizer] = None,
        scheduler_f: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ] = None,
        le: LabelEncoder,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "optim_f",
                "scheduler_f",
            ]
        )
        self.model = model
        self.optim_f = optim_f
        self.scheduler_f = scheduler_f
        self.le = le

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.optim = self.optim_f(self.model)
            self.scheduler = self.scheduler_f(self.optim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y.squeeze().long())
        self.log("loss", loss, prog_bar=True)
        self.log(
            "acc", (y_hat.argmax(dim=1) == y).float().mean(), prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y.squeeze().long())
        self.log("val_loss", loss)
        self.log(
            "val_acc", (y_hat.argmax(dim=1) == y).float().mean(), prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", (y_hat.argmax(dim=1) == y).float().mean())
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return (
            # Return the Truth, Predicted, in that order.
            self.le.inverse_transform(y.cpu().numpy()[..., 0]),
            self.le.inverse_transform(y_hat.argmax(dim=1).cpu().numpy()),
        )

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]
