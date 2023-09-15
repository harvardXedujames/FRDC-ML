from frdc.conf import DATASET_FILE_NAMES
from frdc.evaluate.evaluate import dummy_evaluate
from frdc.load import FRDCDataset
from frdc.preprocess import dummy_preprocess
from frdc.train.train import dummy_train


def test_pipeline():
    ar_im = FRDCDataset()._load_debug_dataset()
    ar_preproc = dummy_preprocess(ar_im[DATASET_FILE_NAMES[0]])
    model = dummy_train(ar_preproc)
    evaluate = dummy_evaluate(model, ar_preproc)
