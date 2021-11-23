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
        # x would be the output of the last layer which is not activated (z)
        # x[i] = W[i] x[i-1] + b[i]
        self.input = input_data
        self.output = self.activator(input_data)

        return self.output

    def backward_propagate(self, output_error, learning_rate):
        return self.activator(self.input, derivative=True) * output_error

    def concatenate(self, last_layer: Layer):
        self.input_shape = self.output_shape = last_layer.output_shape


class Linear(Layer):
    """
        The linear layer whose hypothesis is wx+b.
        ...
        Attributes
        ----------
        units : tuple
            the number of output values of this layer
        input_shape : tuple
            the acceptable input shape of this layer
    """
    def __init__(self, units: int, input_shape: Tuple[int] = None, name: str = None, trainable=True):
        super(Linear, self).__init__(units, input_shape, name, trainable)
        # Usually the input shape is determined by the last layer output shape
        if input_shape is not None:
            self.weights = np.random.randn(*(self.output_shape + self.input_shape))

        self.bias = np.random.randn(*self.output_shape)

    def forward_propagate(self, input_data):
        # the shape of input_data: (batch_size, input_shape)
        self.input = input_data
        self.output = np.array([np.dot(self.weights, x) + self.bias for x in input_data])

        # need to return output to the next layer
        return self.output

    def backward_propagate(self, output_error, learning_rate):
        # shape of output_error: (batch_size, output_dim=units)
        batch_size = output_error.shape[0]
        grad_bias: np.ndarray = np.array([np.dot(self.weights.T, y) for y in output_error])
        grad_weights: np.ndarray = np.array([np.outer(output_error[i], self.input[i]) for i in range(batch_size)])

        # batched gradient descent
        if self.trainable:
            self.weights -= learning_rate * grad_weights.mean(axis=0)
            self.bias -= learning_rate * output_error.mean(axis=0)

        return grad_bias

    def concatenate(self, last_layer: Layer):
        if self.input_shape is None:
            self.input_shape = last_layer.output_shape
            self.weights = np.random.randn(*(self.output_shape + self.input_shape))

        assert self.input_shape == last_layer.output_shape, \
            f'The input shape {self.input_shape} of layer "{self.name}" ' \
            f'does not match the output shape {last_layer.output_shape} of its last layer "{last_layer.name}".'
