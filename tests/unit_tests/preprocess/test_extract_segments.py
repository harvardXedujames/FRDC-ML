""" Test Segment Extraction methods.

We check if the cropping works:
If we crop, then >= 1 shapes in the output must be different from the input.
Else,       then ALL  shapes in the output must be the same as the input.

For example, given that our input image 10 x 10:
- If we crop, then we expect at least one of the output images to be < 10 x 10, for example 7 x 5
- If we don't crop, then we expect all of the output images to be 10 x 10

Thus, our test is:
- If we crop,       any(segment.shape != ar.shape for segment in segments)
- If we don't crop, all(segment.shape == ar.shape for segment in segments)
"""

from frdc.preprocess import extract_segments_from_bounds, extract_segments_from_labels, compute_labels


def test_extract_segments_from_bounds_cropped(ds):
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar := ds.get_ar_bands(), bounds, cropped=True)
    assert any(segment.shape != ar.shape for segment in segments)


def test_extract_segments_from_bounds_no_crop(ds):
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar := ds.get_ar_bands(), bounds, cropped=False)
    assert all(segment.shape == ar.shape for segment in segments)


def test_extract_segments_from_labels_cropped(ds):
    ar_labels = compute_labels(ds.get_ar_bands(), peaks_footprint=10)
    segments = extract_segments_from_labels(ar := ds.get_ar_bands(), ar_labels, cropped=True)
    assert any(segment.shape != ar.shape for segment in segments)


def test_extract_segments_from_labels_no_crop(ds):
    ar_labels = compute_labels(ds.get_ar_bands(), peaks_footprint=10)
    segments = extract_segments_from_labels(ar := ds.get_ar_bands(), ar_labels, cropped=False)
    assert all(segment.shape == ar.shape for segment in segments)
