from frdc.preprocess import extract_segments_from_bounds, extract_segments_from_labels, compute_labels


def test_extract_segments_from_bounds_cropped(ds):
    extract_segments_from_bounds(ds.get_ar_bands(), ds.get_bounds(), cropped=True)


def test_extract_segments_from_bounds_no_crop(ds):
    extract_segments_from_bounds(ds.get_ar_bands(), ds.get_bounds(), cropped=False)


def test_extract_segments_from_labels_cropped(ds):
    ar_labels = compute_labels(ds.get_ar_bands())
    extract_segments_from_labels(ds.get_ar_bands(), ar_labels, cropped=True)


def test_extract_segments_from_labels_no_crop(ds):
    ar_labels = compute_labels(ds.get_ar_bands())
    extract_segments_from_labels(ds.get_ar_bands(), ar_labels, cropped=False)
