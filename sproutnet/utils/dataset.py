import numpy as np
from typing import List


def batches(data: np.ndarray, batch_size: int):
    n = data.shape[0]
    for i in range(0, n, batch_size):
        yield data[i: i+batch_size]


def pad_sequences(data: List, pad_value, max_length: int = -1):
    if max_length < 0:
        max_length = np.max([len(seq) for seq in data])

    for seq in data:
        pad_size = max_length - len(seq)
        if pad_size > 0:
            seq.extend([pad_value] * pad_size)

    return data



