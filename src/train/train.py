import numpy as np


class DummyModel():
    def predict(self, x: np.ndarray) -> float:
        return 0.5


def dummy_train(ar: np.ndarray):
    return DummyModel()
