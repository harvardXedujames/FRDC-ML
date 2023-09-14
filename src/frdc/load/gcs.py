from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from google.cloud import storage
from google.oauth2.service_account import Credentials

from frdc.conf import RSC_DIR, SECRETS_DIR


@dataclass
class GCS:
    credentials: Credentials = None
    rsc_folder: Path = RSC_DIR
    project_id: str = 'frmodel'
    bucket_name: str = 'frdc-scan'
    bucket: storage.Bucket = field(init=False)
    dataset_file_names: tuple[str] = (
        'result_Blue.tif', 'result_Green.tif', 'result_NIR.tif', 'result_Red.tif', 'result_RedEdge.tif',
        'bounds.csv'  # May remove this in the future
    )

    def __post_init__(self):
        # We pull the credentials here instead of the constructor for try-except block to catch the FileNotFoundError
        if self.credentials is None:
            try:
                self.credentials = Credentials.from_service_account_file(next(SECRETS_DIR.glob("*.json")).as_posix())
            except StopIteration:
                raise FileNotFoundError(f"No credentials found in {SECRETS_DIR.as_posix()}")

        client = storage.Client(project=self.project_id, credentials=self.credentials)
        self.bucket = client.bucket(self.bucket_name)

    def list_datasets(self) -> pd.DataFrame:
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
            pd.Series([blob.name for blob in gcs.bucket.list_blobs(match_glob=f"**/{anchor}")])
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
        )

        return df

    def download_dataset(
            self, *,
            survey_site: str, survey_date: str, survey_version: str | None,
            dryrun: bool = True
    ):
        """ Downloads a dataset from Google Cloud Storage.

        Notes:
            Retrieve all valid survey_site, survey_date, survey_version combinations from `list_datasets()`.

        Args:
            survey_site: Survey site name.
            survey_date: Survey date in YYYYMMDD format.
            survey_version: Survey version, can be None.
            dryrun: If True, does not download the dataset, but only prints the files to be downloaded.

        Returns:
            A list of paths to the downloaded files, excluding files failed to download.
        """

        # The directory to the files in the bucket, also locally
        file_dir = f"{survey_site}/{survey_date}/{survey_version + '/' if survey_version else ''}"

        for file_name in self.dataset_file_names:
            # Define full paths to the file in the bucket, also locally
            file_pth = file_dir + file_name
            lcl_file_pth = self.rsc_folder / file_pth
            gcs_file_pth = self.bucket.blob(file_pth)

            if lcl_file_pth.exists():
                print(f"{lcl_file_pth} already exists, skipping...")
                continue

            if gcs_file_pth.exists():
                print(f"Downloading {gcs_file_pth.name} to {lcl_file_pth}...")
                if not dryrun:
                    # Create dir locally
                    lcl_file_pth.parent.mkdir(parents=True, exist_ok=True)
                    # Then download from gcs
                    gcs_file_pth.download_to_filename(lcl_file_pth.as_posix())
            else:
                warnings.warn(f"{gcs_file_pth.name=} not found")

        downloaded_files = list((self.rsc_folder / file_dir).glob("*.tif"))
        return downloaded_files
