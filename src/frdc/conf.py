from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
LOCAL_DATASET_ROOT_DIR = ROOT_DIR / 'rsc'
SECRETS_DIR = ROOT_DIR / '.secrets'
GCS_PROJECT_ID = 'frmodel'
GCS_BUCKET_NAME = 'frdc-scan'


# These are sorted by wavelength
# TODO: Is this a bit ugly? I'm not sure if there's a better way to do this.
class Band:
    BLUE = 0
    GREEN = 1
    RED = 2
    RED_EDGE = 3
    NIR = 4

    FILE_NAMES = (
        'result_Blue.tif',
        'result_Green.tif',
        'result_Red.tif',
        'result_RedEdge.tif',
        'result_NIR.tif'
    )

    BLUE_MAX = 2 ** 14
    GREEN_MAX = 2 ** 14
    RED_MAX = 2 ** 14
    RED_EDGE_MAX = 2 ** 14
    NIR_MAX = 2 ** 14
