from abc import ABC

import cupy as cp
import numpy as np


class Initializer(ABC):
    def __call__(self, shape, **kwargs):
        raise NotImplementedError


class Zeros(Initializer):
    def __call__(self, shape, **kwargs):
        return np.zeros(shape=shape)


class RandomUniform(Initializer):
    def __init__(self, low: float = 0., high: float = 1., seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def __call__(self, shape, **kwargs):
        np.random.seed(self.seed)
        return np.random.uniform(self.low, self.high, size=shape)


class GlorotUniform(Initializer):
    def __init__(self, fan_in: int, fan_out: int, seed=None):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.seed = seed

    def __call__(self, shape, **kwargs):
        limit = np.sqrt(6. / (self.fan_in + self.fan_out))
        np.random.seed(self.seed)
        return np.random.uniform(low=-limit, high=limit, size=shape)


class RandomNormal(Initializer):
    def __init__(self, mean: float = 0., std: float = 1., seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def __call__(self, shape, **kwargs):
        np.random.seed(self.seed)
        return np.random.normal(self.mean, self.std, size=shape)


class Orthogonal(Initializer):
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, **kwargs):
        # assert len(shape) == 2
        assert shape[-2] == shape[-1]

        n_matrix = 1
        for dim in shape[:-2]:
            n_matrix *= dim

        np.random.seed(self.seed)
        M = np.random.randn(n_matrix, *shape[-2:])
        for i in range(n_matrix):
            M[i], R = np.linalg.qr(M[i])

        return M.reshape(shape)
