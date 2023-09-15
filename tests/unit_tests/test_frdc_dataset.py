import numpy as np
import pytest

from frdc.conf import Band
from frdc.load import FRDCDataset


@pytest.fixture(scope='module')
def ds():
    return FRDCDataset()


def test_download_datasets(ds):
    ds.download_datasets(dryrun=True)


def test_download_dataset(ds):
    ds.download_dataset(site='DEBUG', date='0', version=None)


def test_list_datasets(ds):
    df = ds.list_gcs_datasets()
    assert len(df) > 0


def test_load_dataset(ds):
    # Loading the debug dataset indirectly tests the load_dataset method.
    dataset = ds._load_debug_dataset()
    assert dataset.shape[2] == len(Band.FILE_NAMES)
