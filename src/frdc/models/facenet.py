import torch
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

INCEPTION_OUT_DIMS = 1524


class FaceNet(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3,
                               padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=7, kernel_size=3,
                               padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=1)
        self.relu3 = nn.ReLU()

        self.base_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1
        )
        # We remove the last layer by replacing it with an identity layer
        self.base_model.fc = nn.Identity()

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.feature_extraction = nn.Sequential(
            self.conv1, self.relu1,
            self.conv2, self.relu2,
            self.conv3, self.relu3,
            self.base_model
        )

        # Logits & aux_logits are the shape of (batch_size, INCEPTION_OUT_DIMS)
        # Thus concat them to (batch_size, INCEPTION_OUT_DIMS * 2)
        self.fc = nn.Linear(INCEPTION_OUT_DIMS * 2, n_classes)

    def forward(self, x):
        logits, aux_logits = self.feature_extraction(x)
        x = torch.concat([logits, aux_logits], dim=1)
        return self.fc(x)
