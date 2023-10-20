from skimage.morphology import remove_small_objects, remove_small_holes

from frdc.preprocess.morphology import threshold_binary_mask, binary_watershed
from frdc.preprocess.scale import scale_0_1_per_band


def get_labels(ar, order):
    ar = scale_0_1_per_band(ar)
    ar_mask = threshold_binary_mask(ar, order.index("NIR"))
    ar_mask = remove_small_objects(ar_mask, min_size=100, connectivity=2)
    ar_mask = remove_small_holes(ar_mask, area_threshold=100, connectivity=2)
    ar_labels = binary_watershed(ar_mask)
    return ar_labels
