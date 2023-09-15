from frdc.preprocess.preprocess import *


def test_scale_0_1_per_band():
    """ This test makes sure that each band is scaled independently.

    To do so, we create a random array of shape (100, 100, 5).
    When we scale it, we expect the min and max of each band to be 0 and 1, respectively.
    Thus, there should be exactly 5 zeros and 5 ones in the scaled array.
    """
    scaled = scale_0_1_per_band(np.random.random(100 * 100 * 5).reshape((100, 100, 5)))
    assert scaled.shape == (100, 100, 5)
    assert np.sum(scaled == 0) == 5
    assert np.sum(scaled == 1) == 5
