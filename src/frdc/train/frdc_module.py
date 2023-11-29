from typing import Callable

import torch
from lightning import LightningModule
from torch import nn


class FRDCModule(LightningModule):
    def __init__(
        self,
        *,
        model_f: Callable[[], nn.Module],
        optim_f: Callable[[nn.Module], torch.optim.Optimizer],
        scheduler_f: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_f()
        self.optim = optim_f(self.model)
        self.scheduler = scheduler_f(self.optim)

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
        x = batch[0]
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]
