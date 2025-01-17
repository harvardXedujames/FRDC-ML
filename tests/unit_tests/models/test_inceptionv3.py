import pytest
import torch
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from frdc.models.inceptionv3 import InceptionV3MixMatchModule

N_CLASSES = 42
N_CHANNELS = 3
BATCH_SIZE = 2
MIN_SIZE = InceptionV3MixMatchModule.MIN_SIZE


@pytest.fixture(scope="module")
def inceptionv3():
    return InceptionV3MixMatchModule(
        in_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        lr=1e-3,
        x_scaler=StandardScaler(),
        y_encoder=OrdinalEncoder(),
    )


@pytest.mark.parametrize(
    ["batch_size", "channels", "size", "ok"],
    [
        # Well-formed
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE, True],
        # Can be a larger image
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE + 1, True],
        # Cannot have more channels than specified
        [BATCH_SIZE, N_CHANNELS + 1, MIN_SIZE + 1, False],
        # Cannot have a smaller image
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE - 1, False],
    ],
)
def test_inceptionv3_io(inceptionv3, batch_size, channels, size, ok):
    def check(net, x):
        if ok:
            assert net(x).shape == (BATCH_SIZE, N_CLASSES)
        else:
            with pytest.raises(RuntimeError):
                net(x)

    x = torch.rand((batch_size, channels, size, size))

    inceptionv3.train()
    check(inceptionv3, x)
    inceptionv3.eval()
    check(inceptionv3, x)


def test_inception_frozen(inceptionv3):
    """Assert that the base model is frozen, and the rest is trainable."""

    # Assert not all parameters are frozen
    assert (
        sum(p.numel() for p in inceptionv3.parameters() if p.requires_grad) > 0
    )

    # Assert that the inception model is frozen, except the first layer
    assert sum(
        p.numel()
        for p in inceptionv3.inception.parameters()
        if p.requires_grad
    ) == sum(
        p.numel() for p in inceptionv3.inception.Conv2d_1a_3x3.parameters()
    )
