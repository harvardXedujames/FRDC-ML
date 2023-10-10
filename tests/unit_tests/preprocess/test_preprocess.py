from frdc.preprocess import extract_segments_from_labels
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


def test_extract_segments():
    """ Test our segment extraction function.

    To do this, we have a source array and a label array.
    Our function will loop through each unique label in the label array, then mask the source array with that label.

    The following thus should return
    ar_segments = [
        [[[10],[nan]],[[nan],[nan]]],
        [[[nan],[20]],[[nan],[nan]]],
        [[[nan],[nan]],[[30],[nan]]],
        [[[nan],[nan]],[[nan],[40]]],
    ]
    """
    ar_source = np.array(
        [[[10], [20]],
         [[30], [40]]]
    )
    ar_label = np.array(
        [[0, 1],
         [2, 3]]
    )
    ar_segments = extract_segments_from_labels(ar_source, ar_label, cropped=False)
    assert np.isclose(ar_segments[0],np.array([[[10], [np.nan]], [[np.nan], [np.nan]]]), equal_nan=True).all()
    assert np.isclose(ar_segments[1],np.array([[[np.nan], [20]], [[np.nan], [np.nan]]]), equal_nan=True).all()
    assert np.isclose(ar_segments[2],np.array([[[np.nan], [np.nan]], [[30], [np.nan]]]), equal_nan=True).all()
    assert np.isclose(ar_segments[3],np.array([[[np.nan], [np.nan]], [[np.nan], [40]]]), equal_nan=True).all()
