from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from google.cloud import storage
from google.oauth2.service_account import Credentials

from frdc.conf import LOCAL_DATASET_ROOT_DIR, SECRETS_DIR, GCS_PROJECT_ID, GCS_BUCKET_NAME, Band


@dataclass
class FRDCDownloader:
    credentials: Credentials = None
    local_dataset_root_dir: Path = LOCAL_DATASET_ROOT_DIR
    project_id: str = GCS_PROJECT_ID
    bucket_name: str = GCS_BUCKET_NAME
    bucket: storage.Bucket = field(init=False)

    def __post_init__(self):
        # We pull the credentials here instead of the constructor for try-except block to catch the FileNotFoundError
        if self.credentials is None:
            try:
                self.credentials = Credentials.from_service_account_file(next(SECRETS_DIR.glob("*.json")).as_posix())
            except StopIteration:
                raise FileNotFoundError(f"No credentials found in {SECRETS_DIR.as_posix()}")

        client = storage.Client(project=self.project_id, credentials=self.credentials)
        self.bucket = client.bucket(self.bucket_name)

    def list_gcs_datasets(self, anchor=Band.FILE_NAMES[0]) -> pd.DataFrame:
        """ Lists all datasets from Google Cloud Storage.

        Args:
            anchor: The anchor file to find the dataset.
                    This is used to find the dataset. For example, if we want to find the dataset for
                    "chestnut_nature_park/20201218/183deg/result_Red.tif", then we can use "result_Red.tif" as the
                    anchor file.

        Returns:
            A DataFrame of all blobs in the bucket, with columns site, date, version.

        """

        # The anchor file to find the dataset
        # E.g. "result_Red.tif"
        df = (
            # The list of all blobs in the bucket that contains the anchor file
            # E.g. "chestnut_nature_park/20201218/183deg/result_Red.tif"
            pd.Series([blob.name for blob in self.bucket.list_blobs(match_glob=f"**/{anchor}")])
            # Remove the anchor file name
            # E.g. "chestnut_nature_park/20201218/183deg"
            .str.replace(f"/{anchor}", "")
            .rename("dataset_dir")
            .drop_duplicates()
        )

        return df

    def download_file(self, *, path: Path | str, local_exists_ok: bool = True) -> Path:
        """ Downloads a file from Google Cloud Storage. If the file already exists locally, and the hashes match, it
        will not download the file.

        Args:
            path: Path to the file in GCS.
            local_exists_ok: If True, will not raise an error if the file already exists locally and the hashes match.

        Examples:
            If our file in GCS is in gs://frdc-scan/casuarina/20220418/183deg/result_Blue.tif
            then we can download it with:
            >>> download_file(path=Path("casuarina/20220418/183deg/result_Blue.tif"))

        Raises:
            FileNotFoundError: If the file does not exist in GCS.
            FileExistsError: If the file already exists locally and the hashes match.

        Returns:
            The local path to the downloaded file.
        """
        local_path = self.local_dataset_root_dir / path
        gcs_path = path.as_posix() if isinstance(path, Path) else path
        gcs_blob = self.bucket.blob(gcs_path)

        # If not exists in GCS, raise error
        if not gcs_blob.exists():
            raise FileNotFoundError(f"{gcs_path} does not exist in GCS.")

        # If locally exists & hashes match, return False
        if local_path.exists():
            gcs_blob.reload()  # Necessary to get the md5_hash
            gcs_hash = base64.b64decode(gcs_blob.md5_hash).hex()
            local_hash = hashlib.md5(open(local_path, 'rb').read()).hexdigest()
            logging.debug(f"Local hash: {local_hash}, GCS hash: {gcs_hash}")
            if gcs_hash == local_hash:
                if local_exists_ok:
                    # If local_exists_ok, then don't raise
                    return local_path
                else:
                    raise FileExistsError(f"{local_path} already exists and hashes match.")

        # Else, download
        logging.info(f"Downloading {gcs_blob.name} to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        gcs_blob.download_to_filename(local_path.as_posix())
        return local_path


@dataclass
class FRDCDataset:
    site: str
    date: str
    version: str | None
    dl: FRDCDownloader = field(default_factory=FRDCDownloader)

    @staticmethod
    def _load_debug_dataset() -> FRDCDataset:
        """ Loads a debug dataset from Google Cloud Storage.

        Returns:
            A dictionary of the dataset, with keys as the filenames and values as the images.
        """
        return FRDCDataset(site='DEBUG', date='0', version=None)

    @property
    def dataset_dir(self):
        return Path(f"{self.site}/{self.date}/{self.version + '/' if self.version else ''}")

    def get_ar_bands(self, band_names=Band.FILE_NAMES) -> np.ndarray:
        bands_dict = {}
        for band_name in band_names:
            fp = self.dl.download_file(path=self.dataset_dir / band_name)
            ar_im = self._load_image(fp)
            bands_dict[band_name] = ar_im

        # Sort the bands by the order in Band.FILE_NAMES
        return np.stack([bands_dict[band_name] for band_name in Band.FILE_NAMES], axis=-1)

    def get_bounds(self, file_name='bounds.csv') -> list[tuple[int, int, int, int]]:
        fp = self.dl.download_file(path=self.dataset_dir / file_name)
        df = pd.read_csv(fp)
        return [(i.x0, i.y0, i.x1, i.y1) for i in df.itertuples()]

    @staticmethod
    def _load_image(path: Path | str) -> np.ndarray:
        """ Loads an Image from a path.

        Args:
            path: Path to image. pathlib.Path is preferred, but str is also accepted.

        Returns:
            Image as numpy array.
        """

        im = Image.open(Path(path).as_posix())
        return np.array(im)
