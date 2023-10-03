""" Test Segment Extraction methods.

We check if the cropping works:
If we crop, then >= 1 shapes in the output must be different from the input.
Else,       then ALL  shapes in the output must be the same as the input.

For example, given that our input image 10 x 10:
- If we crop, then we expect at least one of the output images to be < 10 x 10,
    for example 7 x 5
- If we don't crop, then we expect all the output images to be 10 x 10

Thus, our test is:
- If we crop,       any(segment.shape != ar.shape for segment in segments)
- If we don't crop, all(segment.shape == ar.shape for segment in segments)
"""
import numpy as np

from frdc.preprocess import (extract_segments_from_bounds,
                             extract_segments_from_labels, compute_labels,
                             remove_small_segments_from_labels)


def test_remove_small_segments_from_labels():
    """ We'll test that it correctly removes the correct segments.

    The test case:
        1 1 2 2 2
        1 1 2 2 2
        3 3 4 4 4
        3 3 4 4 4
        3 3 4 4 4

    For example, if we removed anything smaller than width 3 or height 3,
        then we expect:
        0 0 0 0 0
        0 0 0 0 0
        0 0 4 4 4
        0 0 4 4 4
        0 0 4 4 4

    Then the unique labels should be {0, 4}

    """
    ar = np.zeros((5, 5), dtype=np.uint8)
    ar[0:2, 0:2] = 1
    ar[0:2, 2:5] = 2
    ar[2:5, 0:2] = 3
    ar[2:5, 2:5] = 4

    def test_unique_labels(expected_labels: set, min_height: int = 2,
                           min_width: int = 2):
        """ Tests the unique labels are as expected. """
        assert set(np.unique(
            remove_small_segments_from_labels(ar, min_height=min_height,
                                              min_width=min_width)
        )) == expected_labels

    # 0 represents "removed" labels are relabelled to the background 0.
    test_unique_labels({1, 2, 3, 4}, min_height=2, min_width=2)
    test_unique_labels({0, 3, 4}, min_height=3, min_width=2)
    test_unique_labels({0, 2, 4}, min_height=2, min_width=3)
    test_unique_labels({0, 4}, min_height=3, min_width=3)
    test_unique_labels({0}, min_height=4, min_width=4)


def test_extract_segments_from_bounds_cropped(ds):
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar := ds.get_ar_bands(), bounds,
                                            cropped=True)
    assert any(segment.shape != ar.shape for segment in segments)


def test_extract_segments_from_bounds_no_crop(ds):
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar := ds.get_ar_bands(), bounds,
                                            cropped=False)
    assert all(segment.shape == ar.shape for segment in segments)


def test_extract_segments_from_labels_cropped(ds):
    ar_labels = compute_labels(ds.get_ar_bands(), peaks_footprint=10)
    segments = extract_segments_from_labels(ar := ds.get_ar_bands(), ar_labels,
                                            cropped=True)
    assert any(segment.shape != ar.shape for segment in segments)


def test_extract_segments_from_labels_no_crop(ds):
    ar_labels = compute_labels(ds.get_ar_bands(), peaks_footprint=10)
    segments = extract_segments_from_labels(ar := ds.get_ar_bands(), ar_labels,
                                            cropped=False)
    assert all(segment.shape == ar.shape for segment in segments)
