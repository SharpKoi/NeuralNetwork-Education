import numpy as np

from sproutnet.nn.layers.common import Layer
from sproutnet.nn.initializer import Initializer, GlorotUniform, Zeros


class Linear(Layer):
    """
        The linear layer whose hypothesis is wx+b.
        ...
        Attributes
        ----------
        units : tuple
            the number of output values of this layer
        input_shape : int
            the input dimension of this layer
    """
    def __init__(self, units: int,
                 input_dim: int = None,
                 weights_initializer: Initializer = None,
                 bias_initializer: Initializer = None,
                 name: str = None,
                 trainable=True):
        super(Linear, self).__init__(units, name=name, trainable=trainable)
        self.output_shape = (units,)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer if bias_initializer else Zeros()
        # Usually the input shape is determined by the last layer output shape
        if input_dim:
            if not self.weights_initializer:
                self.weights_initializer = GlorotUniform(input_dim, units)
            self.input_shape = (input_dim,)
            self.weights = self.weights_initializer(shape=(units, input_dim))

        self.bias = self.bias_initializer(shape=self.output_shape)

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
            if not self.weights_initializer:
                self.weights_initializer = GlorotUniform(self.input_shape[-1], self.units)
            self.weights = self.weights_initializer(shape=(self.units, self.input_shape[-1]))

        assert self.input_shape == last_layer.output_shape, \
            f'The input shape {self.input_shape} of layer "{self.name}" ' \
            f'does not match the output shape {last_layer.output_shape} of its last layer "{last_layer.name}".'
