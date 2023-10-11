import pytest
import torch

from frdc.models import FaceNet

N_CLASSES = 42
BATCH_SIZE = 2
MIN_SIZE = FaceNet.MIN_SIZE
CHANNELS = FaceNet.CHANNELS


@pytest.fixture(scope='module')
def face_net():
    return FaceNet(n_classes=N_CLASSES)


@pytest.mark.parametrize(
    ['batch_size', 'channels', 'size', 'ok'],
    [
        [BATCH_SIZE, CHANNELS, MIN_SIZE, True],
        [BATCH_SIZE, CHANNELS, MIN_SIZE + 1, True],
        [BATCH_SIZE, CHANNELS - 1, MIN_SIZE, False],
        [BATCH_SIZE, CHANNELS + 1, MIN_SIZE, False],
        [BATCH_SIZE, CHANNELS, MIN_SIZE - 1, False],
        [BATCH_SIZE, CHANNELS - 1, MIN_SIZE - 1, False],
        [BATCH_SIZE, CHANNELS + 1, MIN_SIZE - 1, False],
        [1, CHANNELS, MIN_SIZE, False],
        [1, CHANNELS, MIN_SIZE + 1, False],
        [1, CHANNELS - 1, MIN_SIZE, False],
        [1, CHANNELS + 1, MIN_SIZE, False],
        [1, CHANNELS, MIN_SIZE - 1, False],
        [1, CHANNELS - 1, MIN_SIZE - 1, False],
        [1, CHANNELS + 1, MIN_SIZE - 1, False],
    ]
)
def test_face_net_io(face_net, batch_size, channels, size, ok):
    x = torch.rand((batch_size, channels, size, size))
    if ok:
        assert face_net(x).shape == (BATCH_SIZE, N_CLASSES)
    else:
        with pytest.raises(ValueError):
            face_net(x)
