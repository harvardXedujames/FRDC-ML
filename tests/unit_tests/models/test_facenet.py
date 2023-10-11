import pytest
import torch

from frdc.models import FaceNet

N_CLASSES = 42
BATCH_SIZE = 2
MIN_SIZE = 299
CHANNELS = 3


@pytest.fixture(scope='module')
def face_net():
    return FaceNet(n_classes=N_CLASSES)


@pytest.mark.parametrize(
    ['channels', 'size', 'ok'],
    [
        [CHANNELS, MIN_SIZE, True],
        [CHANNELS, MIN_SIZE + 1, True],
        [CHANNELS - 1, MIN_SIZE, False],
        [CHANNELS + 1, MIN_SIZE, False],
        [CHANNELS, MIN_SIZE - 1, False],
        [CHANNELS - 1, MIN_SIZE - 1, False],
        [CHANNELS + 1, MIN_SIZE - 1, False],
    ]
)
def test_face_net_io(face_net, channels, size, ok):
    x = torch.rand((BATCH_SIZE, channels, size, size))
    if ok:
        assert face_net(x).shape == (BATCH_SIZE, N_CLASSES)
    else:
        with pytest.raises(Exception):
            face_net(x)


