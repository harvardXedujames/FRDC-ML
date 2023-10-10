import pytest


def test_download_file_exist_ok(dl, debug_file_path):
    fp = dl.download_file(path_glob=debug_file_path, local_exists_ok=True)
    assert fp.exists(), "File doesn't exist after download."
    with pytest.raises(FileExistsError):
        dl.download_file(path_glob=debug_file_path, local_exists_ok=False)


def test_download_multiple_files(dl):
    """ Test that download_file shouldn't support multiple files. """
    with pytest.raises(ValueError):
        dl.download_file(path_glob='**/*.tif')


def test_list_datasets(dl):
    df = dl.list_gcs_datasets()
    assert len(df) > 0
