import torch
from lightning import LightningModule
from torch import nn


class FRDCModule(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('loss', loss, prog_bar=True)
        self.log('acc', (y_hat.argmax(dim=1) == y).float().mean(),
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', (y_hat.argmax(dim=1) == y).float().mean(),
                 prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', (y_hat.argmax(dim=1) == y).float().mean())
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0]
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
