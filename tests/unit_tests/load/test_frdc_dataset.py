import pytest

from frdc.conf import Band


def test_download_file_exist_ok(dl):
    fp = dl.download_file(path_glob=f'DEBUG/0/{Band.FILE_NAME_GLOBS[0]}')
    assert fp.exists()


def test_download_file_exist_not_ok(dl):
    with pytest.raises(FileExistsError):
        dl.download_file(path_glob=f'DEBUG/0/{Band.FILE_NAME_GLOBS[0]}', local_exists_ok=False)


def test_list_datasets(dl):
    df = dl.list_gcs_datasets()
    assert len(df) > 0
