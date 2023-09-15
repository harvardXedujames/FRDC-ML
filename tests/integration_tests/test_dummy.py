import numpy as np

from frdc.load import FRDCDataset
from frdc.preprocess import compute_segments_mask, extract_segments


def test_auto_segmentation_pipeline():
    ar = FRDCDataset()._load_debug_dataset()
    ar_segments_mask = compute_segments_mask(ar)
    ar_segments = extract_segments(ar, ar_segments_mask)


def test_manual_segmentation_pipeline():
    ar = FRDCDataset()._load_debug_dataset()
    # This is a trivial example of manual segmentation, which bins the first band into 4 segments with equal quantiles.
    # In production, this will be loaded from a ground truth mask.
    qs = [0, 0.25, 0.5, 0.75, 1]
    ar_segments_mask = np.digitize(ar[:, :, 0], np.nanquantile(ar[:, :, 0], qs), False) - 1
    ar_segments = extract_segments(ar, ar_segments_mask)

    # assert that the number of labeled pixels is equal to the number of pixels in the segments
    # ! We -1 because digitize implicitly adds a bin for values above the last quantile
    #   Thus the comparison will not be valid for the last segment
    # TODO: Probably get a ground truth mask from the dataset to avoid this hacky code.
    for segment_ix in range(len(qs) - 1):
        mask_pixels_segment = np.sum(ar_segments_mask == segment_ix)
        non_nan_pixels_segment = np.sum(~np.isnan(ar_segments[segment_ix]))
        assert mask_pixels_segment * ar.shape[-1] == non_nan_pixels_segment

def test_pipeline():
    ar = FRDCDataset()._load_debug_dataset()
    ar_segments_mask = compute_segments_mask(ar)
    X = extract_segments(ar, ar_segments_mask)
