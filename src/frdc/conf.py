from collections import OrderedDict
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
LOCAL_DATASET_ROOT_DIR = ROOT_DIR / 'rsc'
SECRETS_DIR = ROOT_DIR / '.secrets'
GCS_PROJECT_ID = 'frmodel'
GCS_BUCKET_NAME = 'frdc-scan'

# These are sorted by wavelength
BAND_CONFIG = OrderedDict({
    'WR': ('*result.tif', lambda x: x[..., 0:1]),
    'WG': ('*result.tif', lambda x: x[..., 1:2]),
    'WB': ('*result.tif', lambda x: x[..., 2:3]),
    'NR': ('result_Red.tif', lambda x: x),
    'NG': ('result_Green.tif', lambda x: x),
    'NB': ('result_Blue.tif', lambda x: x),
    'RE': ('result_RedEdge.tif', lambda x: x),
    'NIR': ('result_NIR.tif', lambda x: x),
})

# DEFAULT_BAND_SCALED_CONFIG = OrderedDict({
#     'WR': ('*result.tif', lambda x: x[..., 0:1] / 2 ** 8),
#     'WG': ('*result.tif', lambda x: x[..., 1:2] / 2 ** 8),
#     'WB': ('*result.tif', lambda x: x[..., 2:3] / 2 ** 8),
#     'NR': ('result_Red.tif', lambda x: x / 2 ** 14),
#     'NG': ('result_Green.tif', lambda x: x / 2 ** 14),
#     'NB': ('result_Blue.tif', lambda x: x / 2 ** 14),
#     'RE': ('result_RedEdge.tif', lambda x: x / 2 ** 14),
#     'NIR': ('result_NIR.tif', lambda x: x / 2 ** 14),
# })
