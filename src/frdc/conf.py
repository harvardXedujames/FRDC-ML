from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
LOCAL_DATASET_ROOT_DIR = ROOT_DIR / 'rsc'
SECRETS_DIR = ROOT_DIR / '.secrets'
GCS_PROJECT_ID = 'frmodel'
GCS_BUCKET_NAME = 'frdc-scan'


# These are sorted by wavelength
# TODO: Is this a bit ugly? I'm not sure if there's a better way to do this.
class Band:
    WIDE_BLUE = 0
    WIDE_GREEN = 1
    WIDE_RED = 2
    BLUE = 3
    GREEN = 4
    RED = 5
    RED_EDGE = 6
    NIR = 7

    FILE_NAME_GLOBS = (
        '*result.tif',
        'result_Blue.tif',
        'result_Green.tif',
        'result_Red.tif',
        'result_RedEdge.tif',
        'result_NIR.tif'
    )

    WIDE_BLUE_MAX = 2 ** 16
    WIDE_GREEN_MAX = 2 ** 16
    WIDE_RED_MAX = 2 ** 16
    BLUE_MAX = 2 ** 14
    GREEN_MAX = 2 ** 14
    RED_MAX = 2 ** 14
    RED_EDGE_MAX = 2 ** 14
    NIR_MAX = 2 ** 14
