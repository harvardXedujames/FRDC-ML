import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed

from frdc.conf import Band


def compute_segments_mask(
        ar: np.ndarray,
        nir_threshold_value=0.5,
        min_crown_size=100,
        min_crown_hole=100,
        connectivity=1,
        peaks_footprint=20,
        watershed_compactness=0.1
) -> np.ndarray:
    """ Automatically segments crowns from an NDArray with a series of image processing operations.

    Args:
        ar: NDArray of shape (H, W, C), where C is the number of bands, C is sorted by Band.FILE_NAMES.
        nir_threshold_value: Threshold value for the NIR band.
        min_crown_size: Minimum crown size in pixels.
        min_crown_hole: Minimum crown hole size in pixels.
        connectivity: Connectivity for morphological operations.
        peaks_footprint: Footprint for peak_local_max.
        watershed_compactness: Compactness for watershed.

    Returns:
        A tuple of (background, crowns), background is the background image and crowns is a list of np.ndarray crowns.
        Background is of shape (H, W, C), where C is the number of bands, C is sorted by Band.FILE_NAMES.
        Crowns is a list of np.ndarray crowns, each crown is of shape (H, W, C).
    """
    # ar = scale_0_1_per_band(ar)
    ar = scale_static_per_band(ar)
    ar_mask = threshold_binary_mask(ar, Band.NIR, nir_threshold_value)
    ar_mask = remove_small_objects(ar_mask, min_size=min_crown_size, connectivity=connectivity)
    ar_mask = remove_small_holes(ar_mask, area_threshold=min_crown_hole, connectivity=connectivity)
    ar_label = binary_watershed(ar_mask, peaks_footprint, watershed_compactness)
    return ar_label


def scale_0_1_per_band(ar: np.ndarray) -> np.ndarray:
    """ Scales an NDArray from 0 to 1 for each band independently

    Args:
        ar: NDArray of shape (H, W, C), where C is the number of bands.
    """
    ar_bands = []
    for band in range(ar.shape[-1]):
        ar_band = ar[:, :, band]
        ar_band = (ar_band - np.nanmin(ar_band)) / (np.nanmax(ar_band) - np.nanmin(ar_band))
        ar_bands.append(ar_band)

    return np.stack(ar_bands, axis=-1)


def scale_static_per_band(ar: np.ndarray) -> np.ndarray:
    """ This scales statically, by their defined maximum   """
    ar = ar.copy()
    ar[:, :, Band.BLUE] /= Band.BLUE_MAX
    ar[:, :, Band.GREEN] /= Band.GREEN_MAX
    ar[:, :, Band.RED] /= Band.RED_MAX
    ar[:, :, Band.RED_EDGE] /= Band.RED_EDGE_MAX
    ar[:, :, Band.NIR] /= Band.NIR_MAX

    return ar


def threshold_binary_mask(ar: np.ndarray, band: Band, threshold_value: float) -> np.ndarray:
    """ Creates a binary mask array from an NDArray by thresholding a band.

    Notes:
         For the band argument, use Band.NIR for the NIR band, Band.RED for the RED band, etc.

    Returns:
        A binary mask array of shape (H, W), True for values above the threshold, False otherwise.
    """
    return ar[:, :, band] > threshold_value


def binary_watershed(ar_mask: np.ndarray, peaks_footprint: int, watershed_compactness: float) -> np.ndarray:
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
    return watershed(image=-ar_watershed_depth,
                     markers=ar_watershed_basins,
                     mask=ar_mask,
                     # watershed_line=True, # Enable this to see the watershed lines
                     compactness=watershed_compactness)


def extract_segments(ar: np.ndarray, ar_segments_mask: np.ndarray) -> list[np.ndarray]:
    """ Extracts segments as a list from a label image.

    Args:
        ar: The source image to extract segments from.
        ar_segments_mask: Segments Image, where each integer value is a segment mask.

    Returns:
        A list of segments, each segment is of shape (H, W, C).

    """
    ar_segments = []
    for segment_ix in range(np.max(ar_segments_mask) + 1):
        ar_segment_mask = np.array(ar_segments_mask == segment_ix)
        ar_segment = ar.copy()
        ar_segment = np.where(ar_segment_mask[..., None], ar_segment, np.nan)
        ar_segments.append(ar_segment)
    return ar_segments


def extract_segments_crop(ar: np.ndarray, ar_segments_mask: np.ndarray) -> list[np.ndarray]:
    """ Extracts segments as a list from a label image.

    Args:
        ar: The source image to extract segments from.
        ar_segments_mask: Segments Image, where each integer value is a segment mask.
        
    Returns:
        A list of cropped segments, each segment is of shape (H, W, C).

    """
    ar_segments = []
    for segment_ix in range(1, np.max(ar_segments_mask) + 1):
        segment = ar_segments_mask == segment_ix
        coords = np.argwhere(segment)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        segment_size = (x1 - x0) * (y1 - y0)
        ar_segments.append(ar[x0:x1, y0:y1])
    return ar_segments
