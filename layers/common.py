from abc import ABC
from typing import Tuple

import numpy as np


class Layer(ABC):
    def __init__(self, units: int, input_shape: Tuple[int] = None, name: str = None, trainable=False):
        self.name = name
        self.is_hidden = False
        self.trainable = trainable

        self.units = units
        self.output_shape = (units,)
        self.input_shape = input_shape

        self.input = self.output = None

    def forward_propagate(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_propagate(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError

    def concatenate(self, last_layer):
        raise NotImplementedError


class Activation(Layer):
    def __init__(self, activator, name: str = None):
        super(Activation, self).__init__(units=0, input_shape=tuple(), name=name, trainable=False)
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
