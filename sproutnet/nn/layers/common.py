from abc import ABC
from typing import Tuple, Callable, Union

import cupy as cp
import numpy as np


class Layer(ABC):
    def __init__(self, units: int, input_shape: Tuple[int] = None, name: str = None, trainable=False):
        self.name = name
        self.is_hidden = False
        self.trainable = trainable

        self.units = units
        self.input_shape = input_shape
        self.output_shape = None

        self.input = self.output = None

    def forward_propagate(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_propagate(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError

    def concatenate(self, last_layer):
        raise NotImplementedError


class Activation(Layer):
    def __init__(self, activator: Callable, name: str = None):
        super(Activation, self).__init__(units=0, name=name, trainable=False)
        self.activator = activator

    def forward_propagate(self, input_data):
        # x would be the output of the last layer which is not activated
        # x[i] = W[i] x[i-1] + b[i]
        self.input = input_data
        self.output = self.activator(input_data)

        return self.output

    def backward_propagate(self, output_error, learning_rate):
        return self.activator(self.input, derivative=True) * output_error

    def concatenate(self, last_layer: Layer):
        self.input_shape = self.output_shape = last_layer.output_shape


class Embedding(Layer):
    def __init__(self,
                 input_length: int,
                 weights: Union[cp.ndarray, np.ndarray],
                 name: str = None,
                 trainable: bool = False):
        super().__init__(units=-1, name=name, trainable=trainable)
        self.input_length = input_length
        self.input_shape = (self.input_length,)
        self.weights = cp.asarray(weights) if cp.cuda.is_available() else np.copy(weights)
        self.embedding_dim = self.weights.shape[-1]
        self.output_shape = (self.input_length, self.embedding_dim)

    def forward_propagate(self, input_data: np.ndarray) -> np.ndarray:
        # (batch_size, id_seq)
        batch_size, input_dim = input_data.shape

        assert input_dim == self.input_length

        self.input = input_data
        self.output = np.empty(shape=(batch_size, input_dim, self.weights.shape[1]))

        for i in range(batch_size):
            weight = self.weights[input_data[i].tolist()]
            if cp.cuda.is_available():
                self.output[i] = weight.get()
            else:
                self.output[i] = weight

        return self.output

    def backward_propagate(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.trainable:
            batch_size = output_error.shape[0]
            grad_weights = cp.zeros(shape=self.weights.shape)
            divisors = cp.zeros(shape=(self.weights.shape[0],))

            for i in range(batch_size):
                grad_weights[self.input[i]] += output_error[i]
                divisors[self.input[i]] += 1

            divisors[divisors == 0] = 1
            grad_weights /= divisors

            self.weights -= learning_rate * grad_weights

            # there's no sense to update the input sequences
        return np.zeros(shape=self.input_shape)

    def concatenate(self, last_layer):
        """In almost all cases, Embedding layer is the first layer."""
        pass
