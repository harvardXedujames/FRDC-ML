from abc import ABC

import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

from frdc.train import FRDCModule


class InceptionV3(FRDCModule, ABC):
    INCEPTION_OUT_DIMS = 2048
    INCEPTION_AUX_DIMS = 1000
    INCEPTION_IN_CHANNELS = 3
    MIN_SIZE = 299

    def __init__(
        self,
        *,
        n_out_classes: int,
        x_scaler: StandardScaler,
        y_encoder: OrdinalEncoder,
    ):
        """Initialize the InceptionV3 model.

        Args:
            n_out_classes: The number of output classes

        Notes:
            - Min input size: 299 x 299.
            - Batch size: >= 2.

            Retrieve these constants in class attributes MIN_SIZE and CHANNELS.
        """

        super().__init__(x_scaler=x_scaler, y_encoder=y_encoder)

        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
        )
        self.inception.fc = nn.Identity()

        # Freeze base model
        for param in self.inception.parameters():
            param.requires_grad = False

        # self.fc = nn.Linear(self.INCEPTION_OUT_DIMS, n_out_classes)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.INCEPTION_OUT_DIMS),
            nn.Linear(self.INCEPTION_OUT_DIMS, self.INCEPTION_OUT_DIMS // 2),
            nn.BatchNorm1d(self.INCEPTION_OUT_DIMS // 2),
            nn.Linear(self.INCEPTION_OUT_DIMS // 2, n_out_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Notes:
            - Min input size: 299 x 299.
            - Batch size: >= 2.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        """

        if (
            any(s == 1 for s in x.shape)
            or x.shape[2] < self.MIN_SIZE
            or x.shape[3] < self.MIN_SIZE
        ):
            raise RuntimeError(
                f"Input shape {x.shape} must adhere to the following:\n"
                f" - No singleton dimensions\n"
                f" - Size >= {self.MIN_SIZE}\n"
            )

        # During training, the auxiliary outputs are used for auxiliary loss,
        # but during testing, only the main output is used.
        if self.training:
            logits, *_ = self.inception(x)
        else:
            logits = self.inception(x)

        return self.fc(logits)
