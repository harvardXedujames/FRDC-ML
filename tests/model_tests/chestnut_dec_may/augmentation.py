import torch
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomVerticalFlip


def augmentation(t: torch.Tensor) -> torch.Tensor:
    """Runs out augmentation on a tensor."""
    t = RandomHorizontalFlip()(t)
    t = RandomVerticalFlip()(t)
    return t
