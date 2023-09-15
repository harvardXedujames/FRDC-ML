from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from google.cloud import storage
from google.oauth2.service_account import Credentials

from frdc.conf import LOCAL_DATASET_ROOT_DIR, DATASET_FILE_NAMES, SECRETS_DIR, GCS_PROJECT_ID, GCS_BUCKET_NAME


@dataclass
class FRDCDataset:
    credentials: Credentials = None
    local_dataset_root_dir: Path = LOCAL_DATASET_ROOT_DIR
    project_id: str = GCS_PROJECT_ID
    bucket_name: str = GCS_BUCKET_NAME
    bucket: storage.Bucket = field(init=False)
    dataset_file_names: tuple[str] = DATASET_FILE_NAMES

    def __post_init__(self):
        # We pull the credentials here instead of the constructor for try-except block to catch the FileNotFoundError
        if self.credentials is None:
            try:
                self.credentials = Credentials.from_service_account_file(next(SECRETS_DIR.glob("*.json")).as_posix())
            except StopIteration:
                raise FileNotFoundError(f"No credentials found in {SECRETS_DIR.as_posix()}")

        client = storage.Client(project=self.project_id, credentials=self.credentials)
        self.bucket = client.bucket(self.bucket_name)

    def list_gcs_datasets(self) -> pd.DataFrame:
        """ Lists all datasets from Google Cloud Storage.

        Returns:
            A DataFrame of all blobs in the bucket, with columns site, date, version.
        """

        # The anchor file to find the dataset
        # E.g. "result_Red.tif"
        anchor = self.dataset_file_names[0]

        df = (
            # The list of all blobs in the bucket that contains the anchor file
            # E.g. "chestnut_nature_park/20201218/183deg/result_Red.tif"
            pd.Series([blob.name for blob in self.bucket.list_blobs(match_glob=f"**/{anchor}")])
            # Remove the anchor file name
            # E.g. "chestnut_nature_park/20201218/183deg"
            .str.replace(f"/{anchor}", "")
            # Split the path into columns
            # E.g. "chestnut_nature_park/20201218/183deg" -> ["chestnut_nature_park", "20201218", "183deg"]
            .str.split("/", expand=True, n=2)
            # Rename the columns
            # E.g. ["site",                 "date",     "version"]
            #      ["chestnut_nature_park", "20201218", "183deg"]
            .rename(columns={0: "site", 1: "date", 2: "version"})
            # Drop any dupes (likely none)
            .drop_duplicates()
            .reset_index(drop=True)
            .set_index(['site', 'date'])
        )

        return df

    def download_dataset(self, *, site: str, date: str, version: str | None, dryrun: bool = True) -> Path | None:
        """ Downloads a dataset from Google Cloud Storage.

        Notes:
            Retrieve all valid site, date, version combinations from `list_gcs_datasets()`.

        Args:
            site: Survey site name.
            date: Survey date in YYYYMMDD format.
            version: Survey version, can be None.
            dryrun: If True, does not download the dataset, but only prints the files to be downloaded.

        Raises:
            FileNotFoundError: If the dataset does not exist in GCS.

        Returns:
            The local dataset directory of the downloaded dataset if successful, else None.
        """

        # The directory to the files in the bucket, also locally
        dataset_dir = self.get_dataset_dir(site, date, version)

        for dataset_file_name in self.dataset_file_names:
            # Define full paths to the file in the bucket, also locally
            # For local path, ROOT   / DATASET DIR / FILENAME
            # For GCS,        BUCKET / DATASET DIR / FILENAME
            local_file_path = self.local_dataset_root_dir / dataset_dir / dataset_file_name
            gcs_file_path = self.bucket.blob(str(dataset_dir / dataset_file_name))

            # Don't download if the file already exists locally
            if local_file_path.exists():
                print(f"{local_file_path} already exists, skipping...")
                continue

            # Else, download from GCS
            if gcs_file_path.exists():
                print(f"Downloading {gcs_file_path.name} to {local_file_path}...")
                if not dryrun:
                    # Create dir locally
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    # Then download from gcs
                    gcs_file_path.download_to_filename(local_file_path.as_posix())
            else:
                raise FileNotFoundError(f"{gcs_file_path} does not exist in GCS.")

        return self.local_dataset_root_dir / dataset_dir

    def download_datasets(self, site_filter: str | list[str] = None, dryrun: bool = True):
        """ Downloads all datasets from Google Cloud Storage.
        
        Args:
            site_filter: If not None, only downloads datasets with the specified site_filter.
            dryrun: If True, does not download the dataset, but only prints the files to be downloaded.
        """

        # Force the filter as a list, so that when .loc[] is called, it returns the site_filter as an index.
        site_filter = [site_filter] if isinstance(site_filter, str) else site_filter

        datasets = self.list_gcs_datasets() \
            if site_filter is None \
            else self.list_gcs_datasets().loc[site_filter]

        for _, args in datasets.reset_index().iterrows():
            self.download_dataset(**args, dryrun=dryrun)

    def load_dataset(self, *, site: str, date: str, version: str | None) -> dict[str, np.ndarray]:
        """ Loads a dataset from Google Cloud Storage.

        Notes:
            Retrieve all valid site, date, version combinations from `list_gcs_datasets()`.
            Will download the dataset if it does not exist locally.

        Args:
            site: Survey site name.
            date: Survey date in YYYYMMDD format.
            version: Survey version, can be None.

        Returns:
            A dictionary of the dataset, with keys as the filenames and values as the images.
        """
        local_dataset_dir = self.download_dataset(site=site, date=date, version=version, dryrun=False)
        return {filename: self.load_image(local_dataset_dir / filename) for filename in self.dataset_file_names}

    def _load_debug_dataset(self) -> dict[str, np.ndarray]:
        """ Loads a debug dataset from Google Cloud Storage.

        Returns:
            A dictionary of the dataset, with keys as the filenames and values as the images.
        """
        return self.load_dataset(site='DEBUG', date='0', version=None)

    @staticmethod
    def get_dataset_dir(site: str, date: str, version: str | None) -> Path:
        """ Formats a dataset directory.

        Args:
            site: Survey site name.
            date: Survey date in YYYYMMDD format.
            version: Survey version, can be None.

        Returns:
            Dataset directory.
        """
        return Path(f"{site}/{date}/{version + '/' if version else ''}")

    @staticmethod
    def load_image(path: Path | str) -> np.ndarray:
        """ Loads an Image from a path.

        Args:
            path: Path to image. pathlib.Path is preferred, but str is also accepted.

        Returns:
            Image as numpy array.
        """

        im = Image.open(Path(path).as_posix())
        return np.array(im)
