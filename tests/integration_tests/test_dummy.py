from pathlib import Path

import numpy as np
from PIL import Image

from frdc.evaluate.evaluate import dummy_evaluate
from frdc.load import load_image
from frdc.preprocess import dummy_preprocess
from frdc.train.train import dummy_train


def test_pipeline():
    ar = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    IMAGE_NAME = Path("test.png")
    Image.fromarray(ar).save(IMAGE_NAME)

    ar_im = load_image(IMAGE_NAME)
    ar_preproc = dummy_preprocess(ar_im)
    model = dummy_train(ar_preproc)
    evaluate = dummy_evaluate(model, ar_preproc)

    IMAGE_NAME.unlink(missing_ok=True)
