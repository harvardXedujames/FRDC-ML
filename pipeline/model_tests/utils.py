from frdc.load import FRDCDataset
from frdc.preprocess.extract_segments import extract_segments_from_bounds

BANDS = ["NB", "NG", "NR", "RE", "NIR"]


def get_dataset(site, date, version):
    ds = FRDCDataset(site=site, date=date, version=version)
    ar, order = ds.get_ar_bands(BANDS)
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar, bounds, cropped=True)
    return segments, labels
