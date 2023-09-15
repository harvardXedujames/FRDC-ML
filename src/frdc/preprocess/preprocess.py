import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed
from sklearn.preprocessing import minmax_scale

from frdc.conf import Band


def compute_crown_masks(
        bands_dict: dict[str, np.ndarray],
        nir_threshold_value=0.5,
        min_crown_size=100,
        min_crown_hole=100,
        connectivity=1,
        peaks_footprint=20,
        watershed_compactness=0.1
):
    ar = stack_bands(bands_dict)
    ar = scale_0_1_per_band(ar)
    ar_mask = threshold_binary_mask(ar, Band.NIR, nir_threshold_value)
    ar_mask = remove_small_objects(ar_mask, min_size=min_crown_size, connectivity=connectivity)
    ar_mask = remove_small_holes(ar_mask, area_threshold=min_crown_hole, connectivity=connectivity)

    # Watershed
    # For watershed, we need:
    #   Image Depth: The distance from the background
    #   Image Basins: The local maxima of the image depth. i.e. points that are the deepest in the image.

    # We can get the image depth by taking the negative euclidean distance transform of the binary mask.
    # This means that lower values are further away from the background.
    ar_watershed_depth = -distance_transform_edt(ar_mask)

    # For basins, we find the basins, by finding the local maxima of the negative image depth.
    ar_watershed_basin_coords = peak_local_max(
        -ar_watershed_depth,
        footprint=np.ones((peaks_footprint, peaks_footprint)),
        min_distance=1,
        exclude_border=0,
        p_norm=2
    )
    ar_watershed_basins = np.zeros(ar_watershed_depth.shape, dtype=bool)
    ar_watershed_basins[tuple(ar_watershed_basin_coords.T)] = True
    ar_watershed_basins, _ = ndimage.label(ar_watershed_basins)

    # TODO: I noticed that low watershed compactness values produces miniblobs, which can be indicative of redundant
    #  crowns. We should investigate this further.
    ar_watershed = watershed(image=-ar_watershed_depth,
                             markers=ar_watershed_basins,
                             mask=ar_mask,
                             watershed_line=True,
                             compactness=watershed_compactness)

    return ar, ar_watershed


def stack_bands(bands_dict: dict[str, np.ndarray]) -> np.ndarray:
    """ Stacks bands into a single image according to DATASET_FILE_NAMES order, which is sorted by wavelength.

    Examples:
        >>> from frdc.load import FRDCDataset
        >>> bands_dict = FRDCDataset().load_dataset(site='DEBUG', date='0', version=None)
        >>> stacked = stack_bands(bands_dict)

    Args:
        bands_dict: Dictionary of bands, with keys as the band names and values as the images.

    Returns:
        Stacked image. Shape is (H, W, C), where C is the number of bands, C is sorted by Band.FILE_NAMES.

    """
    return np.stack([bands_dict[band_name] for band_name in Band.FILE_NAMES], axis=-1)


def scale_0_1_per_band(ar: np.ndarray) -> np.ndarray:
    """ Scales an NDArray from 0 to 1 for each band independently """
    return np.stack([
        # I know we can do fancy projections here, but this is more readable.
        minmax_scale(ar[:, :, band], feature_range=(0, 1))
        for band in range(ar.shape[-1])],
        axis=-1
    )


def threshold_binary_mask(ar: np.ndarray, band: Band, threshold_value: float) -> np.ndarray:
    """ Creates a binary mask array from an NDArray by thresholding a band.

    Notes:
         For the band argument, use Band.NIR for the NIR band, Band.RED for the RED band, etc.

    Returns:
        A binary mask array of shape (H, W), True for values above the threshold, False otherwise.
    """
    return ar[:, :, band] > threshold_value
