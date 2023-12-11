import pytest

from frdc.load.gcs import download, GCSConfig, list_gcs_datasets


def test_download_file_exist_ok(debug_file_path):
    fp = download(fp=debug_file_path, config=GCSConfig(local_exists_ok=True))
    assert fp.exists(), "File doesn't exist after download."

    # Test that it raises an error if we try to download again
    with pytest.raises(FileExistsError):
        download(fp=debug_file_path, config=GCSConfig(local_exists_ok=False))

    # Test that it doesn't raise an error if we set local_exists_ok=True
    download(fp=debug_file_path, config=GCSConfig(local_exists_ok=True))


def test_download_multiple_files():
    """Test that download_file shouldn't support multiple files."""
    with pytest.raises(ValueError):
        download(fp="**/*.tif")


def test_list_datasets():
    df = list_gcs_datasets()
    assert len(df) > 0
