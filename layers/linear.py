from typing import Tuple

import numpy as np

from common import Layer


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
        # shape of input_data: (batch_size, input_shape)
        self.input = input_data
        self.output = np.array([np.dot(self.weights, x) + self.bias for x in input_data])

        # need to return output to the next layer
        return self.output

    def backward_propagate(self, output_error: np.ndarray, learning_rate):
        # shape of output_error: (batch_size, output_dim=units)
        batch_size = output_error.shape[0]

        # the three gradients of J are from the derivation of backward propagation.
        grad_hypothesis: np.ndarray = np.array([np.dot(self.weights.T, y) for y in output_error])
        grad_weights: np.ndarray = np.array([np.outer(output_error[i], self.input[i]) for i in range(batch_size)])
        grad_bias: np.ndarray = output_error

        # batched gradient descent
        if self.trainable:
            self.weights -= learning_rate * grad_weights.mean(axis=0)
            self.bias -= learning_rate * grad_bias.mean(axis=0)

        return grad_hypothesis

    def concatenate(self, last_layer: Layer):
        if self.input_shape is None:
            self.input_shape = last_layer.output_shape
            self.weights = np.random.randn(*(self.output_shape + self.input_shape))

        assert self.input_shape == last_layer.output_shape, \
            f'The input shape {self.input_shape} of layer "{self.name}" ' \
            f'does not match the output shape {last_layer.output_shape} of its last layer "{last_layer.name}".'
