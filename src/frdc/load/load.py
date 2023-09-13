from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path | str) -> np.ndarray:
    """ Loads an Image from a path.

    Args:
        path: Path to image. pathlib.Path is preferred, but str is also accepted.

    Returns:
        Image as numpy array.
    """

    im = Image.open(Path(path).as_posix())
    return np.array(im)
