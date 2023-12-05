from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.nn.parallel
from lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.nn.functional import one_hot
from torchmetrics.functional import accuracy

import mixmatch.utils.interleave


class MixMatchModule(LightningModule):
    def __init__(
        self,
        *,
        x_scaler: StandardScaler,
        y_encoder: OrdinalEncoder,
        n_classes: int = 10,
        sharpen_temp: float = 0.5,
        mix_beta_alpha: float = 0.75,
        ema_lr: float = 0.001,
        interleave: bool = False,
    ):
        """PyTorch Lightning Module for MixMatch

        Notes:
            This performs MixMatch as described in the paper.
            https://arxiv.org/abs/1905.02249

            This module is designed to be used with any model, not only
            the WideResNet model.

            Furthermore, while it's possible to switch datasets, take a look
            at how we implement the CIFAR10DataModule's DataLoaders to see
            how to implement a new dataset.

        Args:
            n_classes: The number of classes in the dataset.
            sharpen_temp: The temperature to use for sharpening.
            mix_beta_alpha: The alpha to use for the beta distribution
                when mixing.
            ema_lr: The learning rate to use for the EMA.
        """

        super().__init__()

        self.x_scaler = x_scaler
        self.y_encoder = y_encoder
        self.n_classes = n_classes
        self.sharpen_temp = sharpen_temp
        self.mix_beta_alpha = mix_beta_alpha
        self.ema_lr = ema_lr
        self.interleave = interleave
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        ...

    @staticmethod
    def loss_unl_scaler(progress: float) -> float:
        return progress * 75

    @staticmethod
    def loss_lbl(lbl_pred: torch.Tensor, lbl: torch.Tensor):
        return F.cross_entropy(lbl_pred, lbl)

    @staticmethod
    def loss_unl(unl_pred: torch.Tensor, unl: torch.Tensor):
        return torch.mean((torch.softmax(unl_pred, dim=1) - unl) ** 2)

    @staticmethod
    def mix_up(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix up the data

        Args:
            x: The data to mix up.
            y: The labels to mix up.
            alpha: The alpha to use for the beta distribution.

        Returns:
            The mixed up data and labels.
        """
        ratio = np.random.beta(alpha, alpha)
        ratio = max(ratio, 1 - ratio)

        shuf_idx = torch.randperm(x.size(0))

        x_mix = ratio * x + (1 - ratio) * x[shuf_idx]
        y_mix = ratio * y + (1 - ratio) * y[shuf_idx]
        return x_mix, y_mix

    @staticmethod
    def sharpen(y: torch.Tensor, temp: float) -> torch.Tensor:
        """Sharpen the predictions by raising them to the power of 1 / temp

        Args:
            y: The predictions to sharpen.
            temp: The temperature to use.

        Returns:
            The probability-normalized sharpened predictions
        """
        y_sharp = y ** (1 / temp)
        # Sharpening will change the sum of the predictions.
        y_sharp /= y_sharp.sum(dim=1, keepdim=True)
        return y_sharp

    def guess_labels(
        self,
        x_unls: list[torch.Tensor],
    ) -> torch.Tensor:
        """Guess labels from the unlabelled data"""
        y_unls: list[torch.Tensor] = [
            torch.softmax(self.ema_model(u), dim=1) for u in x_unls
        ]
        # The sum will sum the tensors in the list,
        # it doesn't reduce the tensors
        y_unl = sum(y_unls) / len(y_unls)
        return y_unl

    @property
    def progress(self):
        # Progress is a linear ramp from 0 to 1 over the course of training.
        return (
            self.global_step / self.trainer.num_training_batches
        ) / self.trainer.max_epochs

    def training_step(self, batch, batch_idx):
        # Progress is a linear ramp from 0 to 1 over the course of training.q
        (x_lbl, y_lbl), x_unls = batch
        y_lbl = one_hot(y_lbl.long(), num_classes=self.n_classes)

        with torch.no_grad():
            y_unl = self.guess_labels(x_unls=x_unls)
            y_unl = self.sharpen(y_unl, self.sharpen_temp)

        x = torch.cat([x_lbl, *x_unls], dim=0)
        y = torch.cat([y_lbl, y_unl, y_unl], dim=0)
        x_mix, y_mix = self.mix_up(x, y, self.mix_beta_alpha)

        if self.interleave:
            # This performs interleaving, see our wiki for details.
            batch_size = x_lbl.shape[0]
            x_mix = list(torch.split(x_mix, batch_size))

            # Interleave to get a consistent Batch Norm Calculation
            x_mix = mixmatch.utils.interleave.interleave(x_mix, batch_size)

            y_mix_pred = [self(x) for x in x_mix]

            # Un-interleave to shuffle back to original order
            y_mix_pred = mixmatch.utils.interleave.interleave(
                y_mix_pred, batch_size
            )

            y_mix_lbl_pred = y_mix_pred[0]
            y_mix_lbl = y_mix[:batch_size]
            y_mix_unl_pred = torch.cat(y_mix_pred[1:], dim=0)
            y_mix_unl = y_mix[batch_size:]
        else:
            batch_size = x_lbl.shape[0]
            y_mix_pred = self(x_mix)
            y_mix_lbl_pred = y_mix_pred[:batch_size]
            y_mix_unl_pred = y_mix_pred[batch_size:]
            y_mix_lbl = y_mix[:batch_size]
            y_mix_unl = y_mix[batch_size:]

        loss_lbl = self.loss_lbl(y_mix_lbl_pred, y_mix_lbl)
        loss_unl = self.loss_unl(y_mix_unl_pred, y_mix_unl)

        loss_unl_scale = self.loss_unl_scaler(progress=self.progress)

        loss = loss_lbl + loss_unl * loss_unl_scale

        self.log("loss_unl_scale", loss_unl_scale, prog_bar=True)
        self.log("train_loss", loss)
        self.log("train_loss_lbl", loss_lbl)
        self.log("train_loss_unl", loss_unl)

        return loss

    # PyTorch Lightning doesn't automatically no_grads the EMA step.
    # It's important to keep this to avoid a memory leak.
    @torch.no_grad()
    def on_after_backward(self) -> None:
        self.ema_updater.update(self.ema_lr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.ema_model(x)
        loss = F.cross_entropy(y_pred, y.long())

        acc = accuracy(
            y_pred, y, task="multiclass", num_classes=y_pred.shape[1]
        )
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.ema_model(x)
        loss = F.cross_entropy(y_pred, y.long())

        acc = accuracy(
            y_pred, y, task="multiclass", num_classes=y_pred.shape[1]
        )
        self.log("test_loss", loss)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def predict_step(self, batch, *args, **kwargs) -> Any:
        x, y = batch
        y_pred = self.ema_model(x)
        y_true_str = self.y_encoder.inverse_transform(
            y.cpu().numpy().reshape(-1, 1)
        )
        y_pred_str = self.y_encoder.inverse_transform(
            y_pred.argmax(dim=1).cpu().numpy().reshape(-1, 1)
        )
        return y_true_str, y_pred_str

    @torch.no_grad()
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """This method is called before any data transfer to the device.

        We leverage this to do some preprocessing on the data.
        Namely, we use the StandardScaler and OrdinalEncoder to transform the
        data.
        """

        if self.training:
            (x_lab, y), x_unl = batch
            xs = [x_lab, *x_unl]

            b, c, h, w = x_lab.shape

            # Move Channel to the last dimension then transform
            xs_ss: list[np.ndarray] = [
                self.x_scaler.transform(x.permute(0, 2, 3, 1).reshape(-1, c))
                for x in xs
            ]

            # Move Channel back to the second dimension
            xs_: list[torch.Tensor] = [
                torch.from_numpy(x_ss.reshape(b, h, w, c))
                .permute(0, 3, 1, 2)
                .float()
                for x_ss in xs_ss
            ]

            # Ordinal Encoder only accepts (n_samples, 1), so we need to do some
            y: tuple[str]
            y_: torch.Tensor = torch.from_numpy(
                self.y_encoder.transform(np.array(y).reshape(-1, 1)).squeeze()
            )

            # Ordinal Encoders can return a np.nan if the value is not in the
            # categories. We will remove that from the batch.
            x_ = xs_[0][~torch.isnan(y_)]
            y_ = y_[~torch.isnan(y_)]

            return (x_, y_.long()), xs_[1:]

        else:
            x, y = batch

            # Standard Scaler only accepts (n_samples, n_features), so we need to
            # do some fancy reshaping.
            # Note that moving dimensions then reshaping is different from just
            # reshaping!
            x: torch.Tensor
            b, c, h, w = x.shape

            # Move Channel to the last dimension then transform
            x_ss: np.ndarray = self.x_scaler.transform(
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
                self.y_encoder.transform(np.array(y).reshape(-1, 1)).squeeze()
            )

            # Ordinal Encoders can return a np.nan if the value is not in the
            # categories. We will remove that from the batch.
            x_ = x_[~torch.isnan(y_)]
            y_ = y_[~torch.isnan(y_)]

            return x_, y_.long()
