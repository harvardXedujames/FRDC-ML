import pytest
import torch

from frdc.models import FaceNet

N_CLASSES = 42
N_CHANNELS = 3
BATCH_SIZE = 2
MIN_SIZE = FaceNet.MIN_SIZE


@pytest.fixture(scope="module")
def facenet():
    return FaceNet(n_out_classes=N_CLASSES)


@pytest.mark.parametrize(
    ["batch_size", "channels", "size", "ok"],
    [
        # Well-formed
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE, True],
        # Can be a larger image
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE + 1, True],
        # Can have more channels
        [BATCH_SIZE, N_CHANNELS + 1, MIN_SIZE + 1, True],
        # Cannot have a smaller image
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE - 1, False],
        # No Singleton Dimension
        [1, N_CHANNELS, MIN_SIZE, False],
        # No Singleton Dimension
        [BATCH_SIZE, 1, MIN_SIZE, False],
    ],
)
def test_facenet_io(facenet, batch_size, channels, size, ok):
    def check(net, x):
        if ok:
            assert net(x).shape == (BATCH_SIZE, N_CLASSES)
        else:
            with pytest.raises(RuntimeError):
                net(x)

    x = torch.rand((batch_size, channels, size, size))

    facenet.train()
    check(facenet, x)
    facenet.eval()
    check(facenet, x)


def test_facenet_frozen(facenet):
    """Assert that the base model is frozen, and the rest is trainable."""
    assert sum(p.numel() for p in facenet.parameters() if p.requires_grad) > 0
    assert (
        sum(
            p.numel()
            for p in facenet.inception.parameters()
            if p.requires_grad
        )
        == 0
    )