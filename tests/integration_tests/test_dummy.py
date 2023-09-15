import numpy as np

from frdc.evaluate.evaluate import dummy_evaluate
from frdc.load import FRDCDataset
from frdc.preprocess import segment_crowns
from frdc.train.train import dummy_train


def test_pipeline():
    ar = FRDCDataset()._load_debug_dataset()
    ar_background, ar_crowns = segment_crowns(ar)
    model = dummy_train(np.stack(ar_crowns))
    evaluate = dummy_evaluate(model, np.stack(ar_crowns))
