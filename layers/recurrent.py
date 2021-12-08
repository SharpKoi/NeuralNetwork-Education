from typing import Tuple

import numpy as np

from activator import sigmoid, tanh
from common import Layer


class LSTMCell:
    def __init__(self,
                 output_dim: int,
                 input_dim: int = -1,
                 activator=sigmoid,
                 cell_activator=tanh,
                 trainable=True):
        self.trainable = trainable
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activator = activator
        self.cell_activator = cell_activator
        # input, forget, cell_stat, output
        if self.input_dim > 0:
            self.weights = {'input': np.random.randn(output_dim, input_dim),
                            'forget': np.random.randn(output_dim, input_dim),
                            'cell': np.random.randn(output_dim, input_dim),
                            'output': np.random.randn(output_dim, input_dim)}

        self.stat_weights = {'input': np.random.randn(output_dim, output_dim),
                             'forget': np.random.randn(output_dim, output_dim),
                             'cell': np.random.randn(output_dim, output_dim),
                             'output': np.random.randn(output_dim, output_dim)}
        self.bias = {'input': np.random.randn(output_dim),
                     'forget': np.random.randn(output_dim),
                     'cell': np.random.randn(output_dim),
                     'output': np.random.randn(output_dim)}

    def forward_pass(self, x, h, c):
        """
        Calculate hidden state and cell state for the next lstm cell.

        :param x: the input data to this cell.
        :param h: the hidden state of the last cell.
        :param c: the cell state of the last cell.
        :return: cell state and hidden state.
        """
        # TODO: Here assume that x and h are both 1D array without batch_size. To check if batched-compute is used.
        input_gate = np.dot(self.weights['input'], x) + np.dot(self.stat_weights['input'], h) + self.bias['input']
        forget_gate = np.dot(self.weights['forget'], x) + np.dot(self.stat_weights['forget'], h) + self.bias['forget']
        cell_input = np.dot(self.weights['cell'], x) + np.dot(self.stat_weights['cell'], h) + self.bias['cell']
        output_gate = np.dot(self.weights['output'], x) + np.dot(self.stat_weights['output'], h) + self.bias['output']

        input_gate = self.activator(input_gate)
        forget_gate = self.activator(forget_gate)
        output_gate = self.activator(output_gate)
        cell_input = self.cell_activator(cell_input)

        cell_state = forget_gate * c + input_gate * cell_input
        hidden_state = output_gate * self.cell_activator(cell_state)

        return cell_state, hidden_state

    def backward_pass(self):
        pass


class LSTM(Layer):
    def __init__(self, units: int, input_shape: Tuple[int] = None, name: str = None, trainable=True):
        super(LSTM, self).__init__(units, input_shape, name, trainable)
        # Usually the input shape is determined by the last layer output shape
        if input_shape is not None:
            self.weights = np.random.randn(*(self.output_shape + self.input_shape))

        self.bias = np.random.randn(*self.output_shape)

    def forward_propagate(self, input_data: np.ndarray) -> np.ndarray:
        # shape of input_data: (batch_size, timesteps, features)

        pass

    def backward_propagate(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        pass

    def concatenate(self, last_layer):
        pass