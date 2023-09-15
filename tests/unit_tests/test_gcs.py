import pytest

from frdc.conf import DATASET_FILE_NAMES
from frdc.load import FRDCDataset


@pytest.fixture(scope='module')
def ds():
    return FRDCDataset()


def test_download_datasets(ds):
    ds.download_datasets(dryrun=True)


def test_list_datasets(ds):
    df = ds.list_gcs_datasets()
    assert len(df) > 0


def test_download_dataset(ds):
    dataset = ds._load_debug_dataset()
    assert len(dataset) == len(DATASET_FILE_NAMES)
