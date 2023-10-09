from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from google.cloud import storage
from google.oauth2.service_account import Credentials

from frdc.conf import LOCAL_DATASET_ROOT_DIR, GCS_PROJECT_ID, \
    GCS_BUCKET_NAME, Band
from frdc.utils import Rect


@dataclass
class FRDCDownloader:
    credentials: Credentials = None
    local_dataset_root_dir: Path = LOCAL_DATASET_ROOT_DIR
    project_id: str = GCS_PROJECT_ID
    bucket_name: str = GCS_BUCKET_NAME
    bucket: storage.Bucket = field(init=False)

    def __post_init__(self):
        """ Initializes the GCS bucket. """
        # If credentials is None, then use the default credentials.
        # Default credentials are set by the environment variable
        # GOOGLE_APPLICATION_CREDENTIALS, see ADC documentation:
        client = storage.Client(project=self.project_id,
                                credentials=self.credentials)
        self.bucket = client.bucket(self.bucket_name)

    def list_gcs_datasets(self, anchor=Band.FILE_NAME_GLOBS[0]) -> pd.DataFrame:
        """ Lists all datasets from Google Cloud Storage.

        Args:
            anchor: The anchor file to find the dataset.
                    This is used to find the dataset. For example, if we want
                    to find the dataset for
                    "chestnut_nature_park/20201218/183deg/result_Red.tif",
                    then we can use "result_Red.tif" as the anchor file.

        Returns:
            An iterator of all blobs that match the anchor file.
        """

        # The anchor file to find the dataset
        # E.g. "result_Red.tif"
        df = (
            # The list of all blobs in the bucket that contains the anchor file
            # E.g. "chestnut_nature_park/20201218/183deg/result_Red.tif"
            pd.Series([blob.name for blob in
                       self.bucket.list_blobs(match_glob=f"**/{anchor}")])
            # Remove the anchor file name
            # E.g. "chestnut_nature_park/20201218/183deg"
            .str.replace(f"/{anchor}", "")
            .rename("dataset_dir")
            .drop_duplicates()
        )

        return df

    def download_file(self, *, path_glob: Path | str,
                      local_exists_ok: bool = True) -> Path:
        """ Downloads a file from Google Cloud Storage. If the file already
            exists locally, and the hashes match, it will not download the file

        Args:
            path_glob: Path Glob to the file in GCS. This must only match one file.
            local_exists_ok: If True, will not raise an error if the file
                already exists locally and the hashes match.

        Examples:
            If our file in GCS is in
            gs://frdc-scan/casuarina/20220418/183deg/result_Blue.tif
            then we can download it with:
            # >>> download_file(
            # >>>     path=Path("casuarina/20220418/183deg/result_Blue.tif")
            # >>> )

        Raises:
            ValueError: If there are multiple blobs that match the path_glob.
            FileNotFoundError: If the file does not exist in GCS.
            FileExistsError: If the file already exists locally and the hashes
                match.

        Returns:
            The local path to the downloaded file.
        """

        # Check if there are multiple blobs that match the path_glob
        gcs_blobs = list(self.bucket.list_blobs(match_glob=Path(path_glob).as_posix()))

        if len(gcs_blobs) > 1:
            raise ValueError(f"Multiple blobs found for {path_glob}: {gcs_blobs}")
        elif len(gcs_blobs) == 0:
            raise FileNotFoundError(f"No blobs found for {path_glob}")

        # Get the local path and the GCS blob
        gcs_blob = gcs_blobs[0]
        local_path = self.local_dataset_root_dir / gcs_blob.name

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
                    raise FileExistsError(
                        f"{local_path} already exists and hashes match.")

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
            A dictionary of the dataset, with keys as the filenames and values
            as the images.
        """
        return FRDCDataset(site='DEBUG', date='0', version=None)

    @property
    def dataset_dir(self):
        return Path(
            f"{self.site}/{self.date}/"
            f"{self.version + '/' if self.version else ''}"
        )

    def get_ar_bands(self, band_globs=Band.FILE_NAME_GLOBS) -> np.ndarray:
        bands_dict = {}
        for band_glob in band_globs:
            fp = self.dl.download_file(path_glob=self.dataset_dir / band_glob)
            ar_im = self._load_image(fp)

            bands_dict[band_glob] = (
                np.expand_dims(ar_im, axis=-1) if ar_im.ndim == 2 else ar_im
            )

        # Sort the bands by the order in Band.FILE_NAMES
        return np.concatenate(
            [bands_dict[band_name] for band_name in Band.FILE_NAME_GLOBS], axis=-1
        )

    def get_bounds_and_labels(self, file_name='bounds.csv') -> (
            tuple)[Iterable[Rect], Iterable[str]]:
        """ Gets the bounds and labels from the bounds.csv file.

        Notes:
            In the context of np.ndarray, to slice with x, y coordinates,
            you need to slice with [y0:y1, x0:x1]. Which is different from the
            bounds.csv file.

        Args:
            file_name: The name of the bounds.csv file.

        Returns:
            A tuple of (bounds, labels), where bounds is a list of
            (x0, y0, x1, y1) and labels is a list of labels.
        """
        fp = self.dl.download_file(path_glob=self.dataset_dir / file_name)
        df = pd.read_csv(fp)
        return ([Rect(i.x0, i.y0, i.x1, i.y1) for i in df.itertuples()],
                df['name'].tolist())

    @staticmethod
    def _load_image(path: Path | str) -> np.ndarray:
        """ Loads an Image from a path.

        Args:
            path: Path to image. pathlib.Path is preferred, but str is also
                accepted.

        Returns:
            Image as numpy array.
        """

        im = Image.open(Path(path).as_posix())
        return np.array(im)
