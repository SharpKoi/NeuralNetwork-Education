from abc import ABC

import numpy as np


class Metric(ABC):
    def __call__(self, y_pred, y_true):
        raise NotImplementedError


class BinaryAccuracy(Metric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        n_data, n_labels = y_true.shape

        y_pred = (y_pred >= self.threshold).astype(int)
        y_diff = np.abs(y_true - y_pred)
        return (n_data - y_diff.sum()) / n_data

