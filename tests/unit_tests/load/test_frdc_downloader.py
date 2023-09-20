import pandas as pd
import pytest

from frdc.conf import Band


def test_download_file_exist_ok(dl):
    fp = dl.download_file(path=f'DEBUG/0/{Band.FILE_NAMES[0]}')
    assert fp.exists()


def test_download_file_exist_not_ok(dl):
    with pytest.raises(FileExistsError):
        dl.download_file(path=f'DEBUG/0/{Band.FILE_NAMES[0]}', local_exists_ok=False)


def test_list_datasets(dl):
    df = dl.list_gcs_datasets()
    assert len(df) > 0


def test_get_ar_bands(ds):
    ar_bands = ds.get_ar_bands()
    assert ar_bands.shape[-1] == len(Band.FILE_NAMES)


def test_get_bounds(ds):
    df_bounds = ds.get_bounds()
    # TODO: Improve test
    assert isinstance(df_bounds, list)
