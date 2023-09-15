from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from google.cloud import storage
from google.oauth2.service_account import Credentials

from frdc.conf import LOCAL_DATASET_ROOT_DIR, SECRETS_DIR, DATASET_FILE_NAMES, GCS_PROJECT_ID, GCS_BUCKET_NAME
from frdc.utils.utils import get_dataset_dir


@dataclass
class GCS:
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
            A DataFrame of all blobs in the bucket, with columns "survey_site", "survey_date", "filename".
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
            # E.g. ["survey_site",           "survey_date", "survey_version"]
            #       ["chestnut_nature_park", "20201218",    "183deg"]
            .rename(columns={0: "survey_site", 1: "survey_date", 2: "survey_version"})
            # Drop any dupes (likely none)
            .drop_duplicates()
            .reset_index(drop=True)
            .set_index(['survey_site', 'survey_date'])
        )

        return df

    def download_dataset(self, *, site: str, date: str, version: str | None, dryrun: bool = True):
        """ Downloads a dataset from Google Cloud Storage.

        Notes:
            Retrieve all valid site, date, version combinations from `list_gcs_datasets()`.

        Args:
            site: Survey site name.
            date: Survey date in YYYYMMDD format.
            version: Survey version, can be None.
            dryrun: If True, does not download the dataset, but only prints the files to be downloaded.
        """

        # The directory to the files in the bucket, also locally
        dataset_dir = get_dataset_dir(site, date, version)

        for dataset_file_name in self.dataset_file_names:
            # Define full paths to the file in the bucket, also locally
            dataset_file_path = dataset_dir + dataset_file_name
            local_file_path = self.local_dataset_root_dir / dataset_file_path
            gcs_file_path = self.bucket.blob(dataset_file_path)

            if local_file_path.exists():
                print(f"{local_file_path} already exists, skipping...")
                continue

            if gcs_file_path.exists():
                print(f"Downloading {gcs_file_path.name} to {local_file_path}...")
                if not dryrun:
                    # Create dir locally
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    # Then download from gcs
                    gcs_file_path.download_to_filename(local_file_path.as_posix())
            else:
                warnings.warn(f"{gcs_file_path.name=} not found")

    def download_datasets(self, site_filter: str | list[str] = None, dryrun: bool = True):
        """ Downloads all datasets from Google Cloud Storage.
        
        Args:
            site_filter: If not None, only downloads datasets with the specified site_filter.
            dryrun: If True, does not download the dataset, but only prints the files to be downloaded.
        """

        # Force the filter as a list, so that when .loc[] is called, it returns the survey_site_filter as an index.
        site_filter = [site_filter] if isinstance(site_filter, str) else site_filter

        datasets = self.list_gcs_datasets() \
            if site_filter is None \
            else self.list_gcs_datasets().loc[site_filter]

        for _, args in datasets.reset_index().iterrows():
            self.download_dataset(**args, dryrun=dryrun)

    def load_dataset(self, *, site: str, date: str, version: str | None, download_if_missing: bool = True):
        ...