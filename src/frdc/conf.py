from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
LOCAL_DATASET_ROOT_DIR = ROOT_DIR / 'rsc'
SECRETS_DIR = ROOT_DIR / '.secrets'
GCS_PROJECT_ID = 'frmodel'
GCS_BUCKET_NAME = 'frdc-scan'

# These are sorted by wavelength
DATASET_FILE_NAMES = (
    'result_Blue.tif',
    'result_Green.tif',
    'result_Red.tif',
    'result_RedEdge.tif',
    'result_NIR.tif'
)
