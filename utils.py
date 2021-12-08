import numpy as np


def batches(data: np.ndarray, batch_size: int):
    n = data.shape[0]
    for i in range(0, n, batch_size):
        yield data[i: i+batch_size]
