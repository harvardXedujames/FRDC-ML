""" Entry point for the whole ML pipeline. """
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate.evaluate import dummy_evaluate
from load import load_image
from preprocess.preprocess import dummy_preprocess
from train.train import dummy_train

ar = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
IMAGE_NAME = Path("test.png")
Image.fromarray(ar).save(IMAGE_NAME)

ar_im = load_image(IMAGE_NAME)
ar_preproc = dummy_preprocess(ar_im)
model = dummy_train(ar_preproc)
evaluate = dummy_evaluate(model, ar_preproc)

IMAGE_NAME.unlink(missing_ok=True)
