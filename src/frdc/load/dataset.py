from __future__ import annotations

import base64
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Callable, Any

import numpy as np
import pandas as pd
from PIL import Image
from google.cloud import storage
from google.oauth2.service_account import Credentials
from torch.utils.data import Dataset, ConcatDataset

from frdc.conf import (
    LOCAL_DATASET_ROOT_DIR,
    GCS_PROJECT_ID,
    GCS_BUCKET_NAME,
    BAND_CONFIG,
)
from frdc.preprocess.extract_segments import extract_segments_from_bounds
from frdc.utils import Rect


@dataclass
class FRDCDownloader:
    credentials: Credentials = None
    local_dataset_root_dir: Path = LOCAL_DATASET_ROOT_DIR
    project_id: str = GCS_PROJECT_ID
    bucket_name: str = GCS_BUCKET_NAME
    bucket: storage.Bucket = field(init=False)

    def __post_init__(self):
        """Initializes the GCS bucket."""
        # If credentials is None, then use the default credentials.
        # Default credentials are set by the environment variable
        # GOOGLE_APPLICATION_CREDENTIALS, see ADC documentation:
        client = storage.Client(
            project=self.project_id, credentials=self.credentials
        )
        self.bucket = client.bucket(self.bucket_name)

    def list_gcs_datasets(self, anchor="result_Red.tif") -> pd.DataFrame:
        """Lists all datasets from Google Cloud Storage.

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
            pd.Series(
                [
                    blob.name
                    for blob in self.bucket.list_blobs(
                        match_glob=f"**/{anchor}"
                    )
                ]
            )
            # Remove the anchor file name
            # E.g. "chestnut_nature_park/20201218/183deg"
            .str.replace(f"/{anchor}", "")
            .rename("dataset_dir")
            .drop_duplicates()
        )

        return df

    def download_file(
        self, *, path_glob: Path | str, local_exists_ok: bool = True
    ) -> Path:
        """Downloads a file from Google Cloud Storage. If the file already
            exists locally, and the hashes match, it will not download the file

        Args:
            path_glob: Path Glob to the file in GCS. This must only match one
                file.
            local_exists_ok: If True, will not raise an error if the file
                already exists locally and the hashes match.

        Examples:
            If our file in GCS is in
            gs://frdc-scan/casuarina/20220418/183deg/result_Blue.tif
            then we can download it with:

            >>> download_file(
            >>>     path=Path("casuarina/20220418/183deg/result_Blue.tif")
            >>> )

        Raises:
            ValueError: If there are multiple blobs that match the path_glob.
            FileNotFoundError: If the file does not exist in GCS.
            FileExistsError: If the file already exists locally and the hashes
                match.

        Returns:
            The local path to the downloaded file.
        """

        # Check if there are multiple blobs that match the path_glob
        gcs_blobs = list(
            self.bucket.list_blobs(match_glob=Path(path_glob).as_posix())
        )

        if len(gcs_blobs) > 1:
            raise ValueError(
                f"Multiple blobs found for {path_glob}: {gcs_blobs}"
            )
        elif len(gcs_blobs) == 0:
            raise FileNotFoundError(f"No blobs found for {path_glob}")

        # Get the local path and the GCS blob
        gcs_blob = gcs_blobs[0]
        local_path = self.local_dataset_root_dir / gcs_blob.name

        # If locally exists & hashes match, return False
        if local_path.exists():
            gcs_blob.reload()  # Necessary to get the md5_hash
            gcs_hash = base64.b64decode(gcs_blob.md5_hash).hex()
            local_hash = hashlib.md5(open(local_path, "rb").read()).hexdigest()
            logging.debug(f"Local hash: {local_hash}, GCS hash: {gcs_hash}")
            if gcs_hash == local_hash:
                if local_exists_ok:
                    # If local_exists_ok, then don't raise
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


@dataclass
class FRDCDataset(Dataset):
    def __init__(
        self,
        site: str,
        date: str,
        version: str | None,
        transform: Callable[[list[np.ndarray]], list[np.ndarray]] = None,
        target_transform: Callable[[list[str]], list[str]] = None,
    ):
        """Initializes the FRDC Dataset.

        Args:
            site: The site of the dataset, e.g. "chestnut_nature_park".
            date: The date of the dataset, e.g. "20201218".
            version: The version of the dataset, e.g. "183deg".
        """
        self.site = site
        self.date = date
        self.version = version

        self.dl = FRDCDownloader()

        self.ar, self.order = self.get_ar_bands()
        bounds, self.targets = self.get_bounds_and_labels()
        self.ar_segments = extract_segments_from_bounds(self.ar, bounds)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ar_segments)

    def __getitem__(self, idx):
        return (
            self.transform(self.ar_segments[idx])
            if self.transform
            else self.ar_segments[idx],
            self.target_transform(self.targets[idx])
            if self.target_transform
            else self.targets[idx],
        )

    @staticmethod
    def _load_debug_dataset() -> FRDCDataset:
        """Loads a debug dataset from Google Cloud Storage.

        Returns:
            A dictionary of the dataset, with keys as the filenames and values
            as the images.
        """
        return FRDCDataset(site="DEBUG", date="0", version=None)

    @property
    def dataset_dir(self):
        return Path(
            f"{self.site}/{self.date}/"
            f"{self.version + '/' if self.version else ''}"
        )

    def get_ar_bands_as_dict(
        self, bands: Iterable[str] = BAND_CONFIG.keys()
    ) -> dict[str, np.ndarray]:
        """Gets the bands from the dataset as a dictionary of (name, image)

        Notes:
            Use get_ar_bands to get the bands as a concatenated numpy array.
            This is used to preserve the bands separately as keys and values.

        Args:
            bands: The bands to get, e.g. ['WB', 'WG', 'WR']. By default, this
                get all bands in BAND_CONFIG.

        Examples:
            >>> get_ar_bands_as_dict(['WB', 'WG', 'WR']])

            Returns

            >>> {'WB': np.ndarray, 'WG': np.ndarray, 'WR': np.ndarray}

        Returns:
            A dictionary of (KeyName, image) pairs.
        """
        d = {}
        fp_cache = {}

        try:
            config = OrderedDict({k: BAND_CONFIG[k] for k in bands})
        except KeyError:
            raise KeyError(
                f"Invalid band name. Valid band names are {BAND_CONFIG.keys()}"
            )

        for name, (glob, transform) in config.items():
            fp = self.dl.download_file(path_glob=self.dataset_dir / glob)

            # We may use the same file multiple times, so we cache it
            if fp in fp_cache:
                logging.debug(f"Cache hit for {fp}, using cached image...")
                im = fp_cache[fp]
            else:
                logging.debug(f"Cache miss for {fp}, loading...")
                im = self._load_image(fp)
                fp_cache[fp] = im

            d[name] = transform(im)

        return d

    def get_ar_bands(
        self,
        bands: Iterable[str] = BAND_CONFIG.keys(),
    ) -> tuple[np.ndarray, list[str]]:
        """Gets the bands as a numpy array, and the band order as a list.

        Notes:
            This is a wrapper around get_bands, concatenating the bands.

        Args:
            bands: The bands to get, e.g. ['WB', 'WG', 'WR']. By default, this
                get all bands in BAND_CONFIG.

        Examples
            >>> get_ar_bands(['WB', 'WG', 'WR'])

            Returns

            >>> (np.ndarray, ['WB', 'WG', 'WR'])

        Returns:
            A tuple of (ar, band_order), where ar is a numpy array of shape
            (H, W, C) and band_order is a list of band names.
        """

        d: dict[str, np.ndarray] = self.get_ar_bands_as_dict(bands)
        return np.concatenate(list(d.values()), axis=-1), list(d.keys())

    def get_bounds_and_labels(
        self, file_name="bounds.csv"
    ) -> tuple[list[Rect], list[str]]:
        """Gets the bounds and labels from the bounds.csv file.

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
        return (
            [Rect(i.x0, i.y0, i.x1, i.y1) for i in df.itertuples()],
            df["name"].tolist(),
        )

    @staticmethod
    def _load_image(path: Path | str) -> np.ndarray:
        """Loads an Image from a path into a 3D numpy array. (H, W, C)

        Notes:
            If the image has only 1 channel, then it will be (H, W, 1) instead

        Args:
            path: Path to image. pathlib.Path is preferred, but str is also
                accepted.

        Returns:
            3D Image as numpy array.
        """

        im = Image.open(Path(path).as_posix())
        ar = np.array(im)
        return np.expand_dims(ar, axis=-1) if ar.ndim == 2 else ar


class FRDCConcatDataset(ConcatDataset):
    def __init__(self, datasets: list[FRDCDataset]):
        super().__init__(datasets)
        self.datasets = datasets
        #

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y

    @property
    def targets(self):
        return [t for ds in self.datasets for t in ds.targets]
