from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TextIO

import pandas as pd
from PIL import Image
from google.cloud import storage
from google.oauth2.service_account import Credentials

from frdc.conf import (
    LOCAL_DATASET_ROOT_DIR,
    GCS_PROJECT_ID,
    GCS_BUCKET_NAME,
)


@dataclass
class GCSConfig:
    credentials: Credentials = None
    local_dir: Path = LOCAL_DATASET_ROOT_DIR
    project_id: str = GCS_PROJECT_ID
    bucket_name: str = GCS_BUCKET_NAME
    local_exists_ok: bool = True
    client: storage.Client = field(init=False)
    bucket: storage.Bucket = field(init=False)

    def __post_init__(self):
        # If credentials is None, then use the default credentials.
        # See the Google ADC documentation on how to set up.
        self.client = storage.Client(
            project=self.project_id, credentials=self.credentials
        )
        self.bucket = self.client.bucket(self.bucket_name)


def download(
    fp: str | Path,
    config: GCSConfig = GCSConfig(),
) -> Path:
    """Downloads a file from Google Cloud Storage. If the file already
        exists locally, and the hashes match, it will not download the file

    Args:
        fp: Path Glob to the file in GCS. This must only match onefile.
        config: The GCSConfig, configures the GCS client, and behaviour.

    Examples:
        If our file in GCS is in
        gs://<BUCKET>/casuarina/20220418/183deg/result_Blue.tif
        then we can download it with:

        >>> download(
        >>>     fp=Path("casuarina/20220418/183deg/result_Blue.tif")
        >>> )

    Raises:
        ValueError: If there are multiple blobs that match the path_glob.
        FileNotFoundError: If the file does not exist in GCS.
        FileExistsError: If the file already exists locally and the hashes
            match.

    Returns:
        The local path to the downloaded file.
    """

    gcs_blobs = list(config.bucket.list_blobs(match_glob=Path(fp).as_posix()))

    if len(gcs_blobs) > 1:
        raise ValueError(f"Multiple blobs found for {fp}: {gcs_blobs}")
    elif len(gcs_blobs) == 0:
        raise FileNotFoundError(f"No blobs found for {fp}")

    # Get the local path and the GCS blob
    gcs_blob = gcs_blobs[0]
    local_path = config.local_dir / gcs_blob.name

    # If locally exists & hashes match, return False
    if local_path.exists():
        gcs_blob.reload()  # Necessary to get the md5_hash
        gcs_hash = base64.b64decode(gcs_blob.md5_hash).hex()
        local_hash = hashlib.md5(open(local_path, "rb").read()).hexdigest()
        logging.debug(f"Local hash: {local_hash}, GCS hash: {gcs_hash}")
        if gcs_hash == local_hash:
            if config.local_exists_ok:
                return local_path
            else:
                raise FileExistsError(
                    f"{local_path} already exists and hashes match."
                )

    # Else, download
    logging.info(f"Downloading {gcs_blob.name} to {local_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gcs_blob.download_to_filename(local_path.as_posix())
    return local_path


def open_file(
    fp: str,
    mode: str = "r",
    config: GCSConfig = GCSConfig(),
) -> TextIO | BytesIO:
    local_fp = download(fp, config)
    return open(local_fp, mode)


def open_image(fp: str) -> Image:
    return Image.open(open_file(fp, "rb"))


def list_gcs_datasets(
    anchor="result_Red.tif",
    config: GCSConfig = GCSConfig(),
) -> pd.DataFrame:
    """Lists all datasets from Google Cloud Storage.

    Args:
        anchor: The anchor file to find the dataset.
                This is used to find the dataset. For example, if we want
                to find the dataset for
                "chestnut_nature_park/20201218/183deg/result_Red.tif",
                then we can use "result_Red.tif" as the anchor file.
        config: The GCSConfig, configures the GCS client, and behaviour.

    Returns:
        An iterator of all blobs that match the anchor file.
    """

    # The anchor file to find the dataset
    # E.g. "result_Red.tif"
    df = (
        # The list of all blobs in the bucket that contains the anchor file
        # E.g. "chestnut_nature_park/20201218/183deg/result_Red.tif"
        pd.Series(
            [
                blob.name
                for blob in config.bucket.list_blobs(match_glob=f"**/{anchor}")
            ]
        )
        # Remove the anchor file name
        # E.g. "chestnut_nature_park/20201218/183deg"
        .str.replace(f"/{anchor}", "")
        .rename("dataset_dir")
        .drop_duplicates()
    )

    return df
