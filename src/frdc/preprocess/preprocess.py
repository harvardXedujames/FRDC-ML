import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def scale_0_1_per_band(ar: np.ndarray) -> np.ndarray:
    """ Scales an NDArray from 0 to 1 for each band independently

    Args:
        ar: NDArray of shape (H, W, C), where C is the number of bands.
    """
    ar_bands = []
    for band in range(ar.shape[-1]):
        ar_band = ar[:, :, band]
        ar_band = ((ar_band - np.nanmin(ar_band)) /
                   (np.nanmax(ar_band) - np.nanmin(ar_band)))
        ar_bands.append(ar_band)

    return np.stack(ar_bands, axis=-1)


# def scale_static_per_band(ar: np.ndarray) -> np.ndarray:
#     """ This scales statically, by their defined maximum   """
#     ar = ar.copy()
#     ar[:, :, Band.BLUE] /= Band.BLUE_MAX
#     ar[:, :, Band.GREEN] /= Band.GREEN_MAX
#     ar[:, :, Band.RED] /= Band.RED_MAX
#     ar[:, :, Band.RED_EDGE] /= Band.RED_EDGE_MAX
#     ar[:, :, Band.NIR] /= Band.NIR_MAX
#
#     return ar


def threshold_binary_mask(ar: np.ndarray,
                          band_ix: int,
                          threshold_value: float = 90 / 256) -> np.ndarray:
    """ Creates a binary mask array from an NDArray by thresholding a band.

    Notes:
         For the band argument, use Band.NIR for the NIR band,
         Band.RED for the RED band, etc.

    Returns:
        A binary mask array of shape (H, W).
        True for values above the threshold, False otherwise.
    """
    return ar[:, :, band_ix] > threshold_value


def binary_watershed(ar_mask: np.ndarray,
                     peaks_footprint: int = 200,
                     watershed_compactness: float = 0) -> np.ndarray:
    """ Watershed segmentation of a binary mask.
    
    Notes:
        This function is used internally by `segment_crowns`.
        
    Args:
        ar_mask: Binary mask array of shape (H, W).
        peaks_footprint: Footprint for peak_local_max.
        watershed_compactness: Compactness for watershed.
        
    Returns:
        A watershed segmentation of the binary mask.
    """

    # Watershed
    # For watershed, we need:
    #   Image Depth: The distance from the background
    #   Image Basins: The local maxima of the image depth.
    #                 i.e. points that are the deepest in the image.

    # The ar distance is the distance from the background.
    ar_distance = distance_transform_edt(ar_mask)

    # We find the basins by the local maxima of the negative image depth.
    ar_watershed_basin_coords = peak_local_max(
        ar_distance,
        footprint=np.ones((peaks_footprint, peaks_footprint)),
        # min_distance=1,
        exclude_border=0,
        # p_norm=2,
        labels=ar_mask
    )
    ar_watershed_basins = np.zeros(ar_distance.shape, dtype=bool)
    ar_watershed_basins[tuple(ar_watershed_basin_coords.T)] = True
    ar_watershed_basins, _ = ndimage.label(ar_watershed_basins)

    # TODO: I noticed that low watershed compactness values produces miniblobs,
    #  which can be indicative of redundant crowns.
    #  We should investigate this further.
    return watershed(image=-ar_distance,
                     # We use the negative so that "peaks" become "troughs"
                     markers=ar_watershed_basins,
                     mask=ar_mask,
                     # Enable this to see the watershed lines
                     # watershed_line=True,
                     compactness=watershed_compactness
                     )
