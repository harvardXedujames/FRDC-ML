""" This suite of tests tests for researcher use case compatibility. """
from frdc.load import FRDCDataset
from frdc.preprocess.extract_segments import extract_segments_from_bounds


def test_load_dataset_and_slice_bounds():
    """ Test the load_dataset method. """
    dataset = FRDCDataset._load_debug_dataset()
    df_bounds = dataset.get_bounds()

