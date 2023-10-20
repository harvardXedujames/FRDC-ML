import numpy as np
import pytest

from frdc.preprocess.scale import scale_0_1_per_band, scale_normal_per_band


def test_scale_0_1_per_band():
    """This test makes sure that each band is scaled independently.

    To do so, we create a random array of shape (100, 100, 5).
    When we scale it, we expect the min and max of each band to be 0 and 1,
     respectively.
    Thus, there should be exactly 5 zeros and 5 ones in the scaled array.
    """
    scaled = scale_0_1_per_band(
        np.random.random(100 * 100 * 5).reshape((100, 100, 5))
    )
    assert scaled.shape == (100, 100, 5)
    assert np.sum(scaled == 0) == 5
    assert np.sum(scaled == 1) == 5


@pytest.mark.parametrize("epsilon", [True, 1e-7])
def test_scale_0_1_per_band_epsilon(epsilon):
    """This test ensures that our epsilon prevents the upper bound from being
    1.0.
    """
    scaled = scale_0_1_per_band(
        np.random.random(100 * 100 * 5).reshape((100, 100, 5)), epsilon=epsilon
    )
    assert scaled.shape == (100, 100, 5)
    assert np.sum(scaled == 0) == 5
    assert np.sum(scaled == 1) == 0


def test_scale_normal_per_band():
    """Ensure that each band is scaled independently, has zero mean and
    unit variance."""

    scaled = scale_normal_per_band(
        np.random.random(100 * 100 * 5).reshape((100, 100, 5))
    )
    assert scaled.shape == (100, 100, 5)
    for band in range(5):
        assert np.isclose(np.mean(scaled[..., band]), 0)
        assert np.isclose(np.std(scaled[..., band]), 1)
