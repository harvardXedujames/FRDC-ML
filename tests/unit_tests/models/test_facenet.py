import pytest
import torch

from frdc.models import FaceNet

N_CLASSES = 42
N_CHANNELS = 2
BATCH_SIZE = 2
MIN_SIZE = FaceNet.MIN_SIZE


@pytest.fixture(scope='module')
def face_net():
    return FaceNet(n_in_channels=N_CHANNELS, n_out_classes=N_CLASSES)


@pytest.mark.parametrize(
    ['batch_size', 'channels', 'size', 'ok'],
    [
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE, True],
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE + 1, True],
        [BATCH_SIZE, N_CHANNELS - 1, MIN_SIZE, False],
        [BATCH_SIZE, N_CHANNELS + 1, MIN_SIZE, False],
        [BATCH_SIZE, N_CHANNELS, MIN_SIZE - 1, False],
        [BATCH_SIZE, N_CHANNELS - 1, MIN_SIZE - 1, False],
        [BATCH_SIZE, N_CHANNELS + 1, MIN_SIZE - 1, False],
        [1, N_CHANNELS, MIN_SIZE, False],
        [1, N_CHANNELS, MIN_SIZE + 1, False],
        [1, N_CHANNELS - 1, MIN_SIZE, False],
        [1, N_CHANNELS + 1, MIN_SIZE, False],
        [1, N_CHANNELS, MIN_SIZE - 1, False],
        [1, N_CHANNELS - 1, MIN_SIZE - 1, False],
        [1, N_CHANNELS + 1, MIN_SIZE - 1, False],
    ]
)
def test_face_net_io(face_net, batch_size, channels, size, ok):
    x = torch.rand((batch_size, channels, size, size))
    if ok:
        assert face_net(x).shape == (BATCH_SIZE, N_CLASSES)
    else:
        with pytest.raises(RuntimeError):
            face_net(x)
