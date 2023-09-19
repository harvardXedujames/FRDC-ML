import numpy as np
import pytest

from frdc.conf import Band
from frdc.load import FRDCDownloader


@pytest.fixture(scope='module')
def dl():
    return FRDCDownloader()


def test_download_datasets(dl):
    dl.download_datasets(dryrun=True)


def test_download_dataset(dl):
    dl.download_dataset(site='DEBUG', date='0', version=None)


def test_list_datasets(dl):
    df = dl.list_gcs_datasets()
    assert len(df) > 0


def test_load_dataset(dl):
    # Loading the debug dataset indirectly tests the load_dataset method.
    ds = dl._load_debug_dataset()
    ar = ds.ar_bands
    assert ar.shape[2] == len(Band.FILE_NAMES)
