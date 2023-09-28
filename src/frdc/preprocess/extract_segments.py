from typing import Iterable

import numpy as np



def remove_small_segments_from_labels(
        ar_labels: np.ndarray,
        min_height: int = 10,
        min_width: int = 10
) -> np.ndarray:
    """ Removes small segments from a label image.

    Args:
        ar_labels: Labelled Image, where each integer value is a segment mask.
        min_height: Minimum height of a segment to be considered "small".
        min_width: Minimum width of a segment to be considered "small".

    Returns:
        A labelled image with small segments removed.
    """
    ar_labels = ar_labels.copy()
    for i in np.unique(ar_labels):
        coords = np.argwhere(ar_labels == i)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        height = y1 - y0
        width = x1 - x0
        if height < min_height or width < min_width:
            logging.info(f"Removing segment {i} with shape {height}x{width}")
            ar_labels[ar_labels == i] = 0
    return ar_labels


def extract_segments_from_labels(
        ar: np.ndarray,
        ar_labels: np.ndarray,
        cropped: bool = True
) -> list[np.ndarray]:
    """ Extracts segments as a list from a label image.

    Args:
        ar: The source image to extract segments from.
        ar_labels: Labelled Image, where each integer value is a segment mask.
        cropped: Whether to crop the segments to the smallest possible size.

    Returns:
        A list of segments, each segment is of shape (H, W, C).

    """
    ar_segments = []
    for segment_ix in range(np.max(ar_labels) + 1):
        ar_segment_mask = np.array(ar_labels == segment_ix)
        if cropped:
            coords = np.argwhere(ar_segment_mask)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1
            ar_segments.append(ar[x0:x1, y0:y1])
        else:
            ar_segment = ar.copy()
            ar_segment = np.where(ar_segment_mask[..., None], ar_segment, np.nan)
            ar_segments.append(ar_segment)
    return ar_segments


def extract_segments_from_bounds(
        ar: np.ndarray,
        bounds: Iterable[Iterable[int]],
        cropped: bool = True
) -> list[np.ndarray]:
    """ Extracts segments as a list from bounds

    Args:
        ar: The source image to extract segments from.
        bounds: The bounds of the segment to extract. (x0, y0, x1, y1)
        cropped: Whether to crop the segments to the smallest possible size.

    Returns:
        A list of segments, each segment is of shape (H, W, C).

    """
    ar_segments = []
    for x0, y0, x1, y1 in bounds:
        if cropped:
            ar_segments.append(ar[x0:x1, y0:y1])
        else:
            ar_segment_mask = np.zeros(ar.shape[:2], dtype=bool)
            ar_segment_mask[y0:y1, x0:x1] = True
            ar_segment = np.where(ar_segment_mask[..., None], ar, np.nan)
            ar_segments.append(ar_segment)
    return ar_segments
