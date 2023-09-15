def get_dataset_dir(site: str, date: str, version: str | None):
    """ Formats a dataset directory.

    Args:
        site: Survey site name.
        date: Survey date in YYYYMMDD format.
        version: Survey version, can be None.

    Returns:
        Dataset directory.
    """
    return f"{site}/{date}/{version + '/' if version else ''}"
