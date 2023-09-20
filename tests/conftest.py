import pytest

from frdc.load import FRDCDownloader, FRDCDataset


@pytest.fixture(scope='session')
def dl() -> FRDCDownloader:
    return FRDCDownloader()


@pytest.fixture(scope='session')
def ds() -> FRDCDataset:
    return FRDCDataset._load_debug_dataset()
