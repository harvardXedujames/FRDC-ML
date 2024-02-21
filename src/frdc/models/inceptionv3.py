from copy import deepcopy

import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.inception import BasicConv2d, Inception3

from frdc.train.mixmatch_module import MixMatchModule
from frdc.utils.ema import EMA


class InceptionV3MixMatchModule(MixMatchModule):
    INCEPTION_OUT_DIMS = 2048
    INCEPTION_AUX_DIMS = 1000
    INCEPTION_IN_CHANNELS = 3
    MIN_SIZE = 299

    def __init__(
        self,
        *,
        in_channels: int,
        n_classes: int,
        lr: float,
        x_scaler: StandardScaler,
        y_encoder: OrdinalEncoder,
        ema_lr: float = 0.001,
        imagenet_scaling: bool = False,
    ):
        """Initialize the InceptionV3 model.

        Args:
            in_channels: The number of input channels.
            n_classes: The number of classes.
            lr: The learning rate.
            x_scaler: The X input StandardScaler.
            y_encoder: The Y input OrdinalEncoder.
            ema_lr: The learning rate for the EMA model.
            imagenet_scaling: Whether to use the adapted ImageNet scaling.

        Notes:
            - Min input size: 299 x 299.
            - Batch size: >= 2.

            Retrieve these constants in class attributes MIN_SIZE and CHANNELS.
        """
        self.lr = lr

        super().__init__(
            n_classes=n_classes,
            x_scaler=x_scaler,
            y_encoder=y_encoder,
            sharpen_temp=0.5,
            mix_beta_alpha=0.75,
        )

        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )

        # Remove the final layer
        self.inception.fc = nn.Identity()

        # Freeze inception weights
        for param in self.inception.parameters():
            param.requires_grad = False

        # Adapt the first layer to accept the number of channels
        self.inception = self.adapt_inception_multi_channel(
            self.inception, in_channels
        )

        self.fc = nn.Sequential(
            nn.Linear(self.INCEPTION_OUT_DIMS, self.INCEPTION_OUT_DIMS // 2),
            nn.BatchNorm1d(self.INCEPTION_OUT_DIMS // 2),
            nn.Linear(self.INCEPTION_OUT_DIMS // 2, n_classes),
            nn.Softmax(dim=1),
        )

        # The problem is that the deep copy runs even before the module is
        # initialized, which means ema_model is empty.
        ema_model = deepcopy(self)
        for param in ema_model.parameters():
            param.detach_()

        self._ema_model = ema_model
        self.ema_updater = EMA(model=self, ema_model=self.ema_model)
        self.ema_lr = ema_lr

    @staticmethod
    def adapt_inception_multi_channel(
        inception: Inception3,
        in_channels: int,
    ) -> Inception3:
        """Adapt the 1st layer of the InceptionV3 model to accept n-channels.

        Notes:
            This operation is in-place, however will still return the model

        Args:
            inception: The InceptionV3 model
            in_channels: The number of input channels

        Returns:
            The adapted InceptionV3 model.
        """

        original_in_channels = inception.Conv2d_1a_3x3.conv.in_channels

        # Replicate the first layer, but with a different number of channels
        conv2d_1a_3x3 = BasicConv2d(
            in_channels=in_channels,
            out_channels=inception.Conv2d_1a_3x3.conv.out_channels,
            kernel_size=inception.Conv2d_1a_3x3.conv.kernel_size,
            stride=inception.Conv2d_1a_3x3.conv.stride,
        )

        # Copy the BGR weights from the first layer of the original model
        conv2d_1a_3x3.conv.weight.data[
            :, :original_in_channels
        ] = inception.Conv2d_1a_3x3.conv.weight.data

        # We'll repeat the G weights to the other channels as an initial
        # approximation
        # We use [1:2] instead of [1] so it doesn't lose the dimension
        conv2d_1a_3x3.conv.weight.data[
            :, original_in_channels:
        ] = inception.Conv2d_1a_3x3.conv.weight.data[:, 1:2].tile(
            (in_channels - original_in_channels, 1, 1)
        )

        # Finally, set the new layer back
        inception.Conv2d_1a_3x3 = conv2d_1a_3x3

        return inception

    @staticmethod
    def imagenet_scaling(x: torch.Tensor) -> torch.Tensor:
        """Perform adapted ImageNet normalization on the input tensor.

        See Also:
            torchvision.models.inception.Inception3._transform_input

        Notes:
            This is adapted from the original InceptionV3 model, which
            uses an RGB transformation. We have adapted it to accept
            any number of channels.

            Additional channels will use the same mean and std as the
            green channel. This is because our task-domain is green-dominant.

        """
        x_ch0 = (
            torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        )
        x_ch1 = (
            torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        )
        x_ch2 = (
            torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        )
        x_chk = x[:, 3:] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2, x_chk), 1)
        return x

    @property
    def ema_model(self):
        return self._ema_model

    def update_ema(self):
        self.ema_updater.update(self.ema_lr)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Notes:
            Min input size: 299 x 299.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        """

        if x.shape[2] < self.MIN_SIZE or x.shape[3] < self.MIN_SIZE:
            raise RuntimeError(
                f"Input shape {x.shape} is too small for InceptionV3.\n"
                f"Minimum size: {self.MIN_SIZE} x {self.MIN_SIZE}.\n"
                f"Got: {x.shape[2]} x {x.shape[3]}."
            )

        if self.imagenet_scaling:
            x = self.imagenet_scaling(x)

        # During training, the auxiliary outputs are used for auxiliary loss,
        # but during testing, only the main output is used.
        if self.training:
            logits, *_ = self.inception(x)
        else:
            logits = self.inception(x)

        return self.fc(logits)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )
