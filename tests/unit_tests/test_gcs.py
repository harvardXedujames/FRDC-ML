from frdc.load import GCS


def test_gcs_download_datasets():
    gcs = GCS()
    gcs.download_datasets(dryrun=True)


def test_gcs_list_datasets():
    gcs = GCS()
    df = gcs.list_gcs_datasets()
    assert len(df) > 0


def test_gcs_download_dataset():
    gcs = GCS()
    gcs.download_dataset(survey_site='chestnut_nature_park', survey_date='20201218', survey_version=None, dryrun=True)
