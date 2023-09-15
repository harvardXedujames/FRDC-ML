import numpy as np

from frdc.evaluate.evaluate import dummy_evaluate
from frdc.load import FRDCDataset
from frdc.preprocess import segment_crowns
from frdc.train.train import dummy_train


def test_pipeline():
    bands_dict = FRDCDataset()._load_debug_dataset()
    ar_background, ar_crowns = segment_crowns(bands_dict)
    model = dummy_train(np.stack(ar_crowns))
    evaluate = dummy_evaluate(model, np.stack(ar_crowns))
