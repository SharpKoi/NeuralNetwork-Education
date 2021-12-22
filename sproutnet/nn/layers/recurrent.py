from typing import Tuple, Callable, Optional

import numpy as np

from sproutnet.nn.layers.common import Layer
from sproutnet.nn.activator import sigmoid, tanh
from sproutnet.nn.initializer import Initializer, GlorotUniform, Orthogonal, Zeros


class LSTM(Layer):
    """
    LSTM layer.

    The output dimension is determined by units and the count of cells in this layer is determined by input dimension.

    ...

    Attributes
    ----------
    units: int
        the output dim of this layer
    input_shape: Tuple[int]
        the input shape without batch size thus should be 2d
    activator: Any
        the activator function used to activate i, f, o in each cell
    cell_activator: Any
        the activator function used to activate c in each cell
    return_sequence: bool
        return the outputs of all the lstm cells or not
    truncate_size: int
        the timesteps TBPTT takes
    """

    def __init__(self,
                 units: int,
                 input_shape: Optional[Tuple[int, int]] = None,
                 activator: Callable = sigmoid,
                 cell_activator: Callable = tanh,
                 weights_initializer: Initializer = None,
                 state_weights_initializer: Initializer = None,
                 bias_initializer: Initializer = None,
                 return_sequence: bool = False,
                 truncate_size: int = -1,
                 name: str = None,
                 trainable: bool = True):
        super(LSTM, self).__init__(units, input_shape, name, trainable)
        self.units = units
        self.activator = activator
        self.cell_activator = cell_activator
        self.weights_initializer = weights_initializer
        self.state_weights_initializer = state_weights_initializer if state_weights_initializer else Orthogonal()
        self.bias_initializer = bias_initializer if bias_initializer else Zeros()
        self.return_sequence = return_sequence
        self.truncate_size = truncate_size
        if input_shape:
            self.input_shape = input_shape
            self.timesteps, self.input_dim = input_shape
            self.output_shape = (self.timesteps, self.units) if self.return_sequence else (self.units,)
            self.truncate_size = self.timesteps if truncate_size <= 0 else min(truncate_size, self.timesteps)
            # i, f, o, c
            if self.input_dim > 0:
                if not self.weights_initializer:
                    self.weights_initializer = GlorotUniform(self.input_dim, self.units)
                self.weights = weights_initializer(shape=(4, self.units, self.input_dim))

        self.state_weights = self.state_weights_initializer(shape=(4, self.units, self.units))
        self.bias = self.bias_initializer(shape=(4, self.units))

        # declare gates and signals
        self.cache_input = None
        self.cache_forget = None
        self.cache_output = None
        self.cache_cell = None
        self.cell_states = None
        self.hidden_states = None

    def forward_propagate(self, input_data: np.ndarray) -> np.ndarray:
        # shape of input_data: (batch_size, timesteps, input_dim)
        batch_size, n_timesteps, n_features = input_data.shape

        assert n_timesteps == self.timesteps and n_features == self.input_dim

        self.input = input_data

        zeros = np.zeros(shape=(batch_size, self.timesteps+1, self.units))
        self.cache_input = np.copy(zeros)
        self.cache_forget = np.copy(zeros)
        self.cache_output = np.copy(zeros)
        self.cache_cell = np.copy(zeros)
        self.cell_states = np.copy(zeros)
        self.hidden_states = np.copy(zeros)

        for i in range(batch_size):
            # shift 1 to preserve zeros at the first.
            for t in range(1, self.timesteps+1):
                # get x_t, h_t-1, c_t-1
                x = input_data[i][t-1]
                h = self.hidden_states[i][t-1]
                c = self.cell_states[i][t-1]

                # ====================== start computing feedforward ====================== #
                self.cache_input[i][t] = np.dot(self.weights[0], x) + np.dot(self.state_weights[0], h) + self.bias[0]
                self.cache_forget[i][t] = np.dot(self.weights[1], x) + np.dot(self.state_weights[1], h) + self.bias[1]
                self.cache_output[i][t] = np.dot(self.weights[2], x) + np.dot(self.state_weights[2], h) + self.bias[2]
                self.cache_cell[i][t] = np.dot(self.weights[3], x) + np.dot(self.state_weights[3], h) + self.bias[3]

                input_gate = self.activator(self.cache_input[i][t])
                forget_gate = self.activator(self.cache_forget[i][t])
                output_gate = self.activator(self.cache_output[i][t])
                cell_input = self.cell_activator(self.cache_cell[i][t])

                self.cell_states[i][t] = forget_gate * c + input_gate * cell_input
                self.hidden_states[i][t] = output_gate * self.cell_activator(self.cell_states[i][t])
                # ====================== end of feedforward ====================== #

        if self.return_sequence:
            self.output = self.hidden_states
        else:
            self.output = self.hidden_states[:, -1]

        return self.output

    def backward_propagate(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        # output_error (batch_size, units)
        batch_size = output_error.shape[0]

        grad_gates = np.zeros(shape=(4, batch_size, self.units))
        grad_weights = np.zeros(shape=(4, batch_size, *self.weights.shape[1:]))
        grad_state_weights = np.zeros(shape=(4, batch_size, *self.state_weights.shape[1:]))
        grad_bias = np.zeros(shape=(4, batch_size, *self.bias.shape[1:]))
        grad_hypothesis = np.zeros(shape=self.input.shape)

        truncated = False
        # ====================== starting gradient computation ====================== #
        for t in reversed(range(1, self.timesteps+1)):
            grad_cell_state = output_error * \
                              self.activator(self.cache_output[:, t, :]) * \
                              self.cell_activator(self.cell_states[:, t, :], derivative=True)
            # i
            grad_gates[0] = grad_cell_state * \
                            self.cache_cell[:, t, :] * \
                            self.activator(self.cache_input[:, t, :], derivative=True)
            # f
            grad_gates[1] = grad_cell_state * \
                            self.cell_states[:, t - 1] * \
                            self.activator(self.cache_forget[:, t, :], derivative=True)
            # o
            grad_gates[2] = output_error * \
                            self.cell_activator(self.cell_states[:, t, :]) * \
                            self.activator(self.cache_output[:, t, :], derivative=True)
            # c
            grad_gates[3] = grad_cell_state * \
                            self.activator(self.cache_input[:, t, :]) * \
                            self.activator(self.cache_cell[:, t, :], derivative=True)

            # totally 4 gates: i, f, o, c
            for j in range(4):
                # compute the next output error
                output_error += np.matmul(grad_gates[j], self.state_weights[j].T)

                # update gradients only if not truncated
                if not truncated:
                    for i in range(batch_size):
                        grad_weights[j][i] += np.outer(grad_gates[j][i], self.input[i][t-1])
                        grad_state_weights[j][i] += np.outer(grad_gates[j][i], self.hidden_states[i, t-1, :])
                    grad_bias[j] += grad_gates[j]

                # (batch_size, input_dim) <- (batch_size, units)(units, input_dim)
                grad_hypothesis[:, t-1, :] += np.matmul(grad_gates[j], self.weights[j])

            truncated = ((self.timesteps - t - 1) % self.truncate_size) == 0
        # ====================== end of gradient computation ====================== #

        # do BPTT
        self.weights -= learning_rate * grad_weights.mean(axis=1)
        self.state_weights -= learning_rate * grad_state_weights.mean(axis=1)
        self.bias -= learning_rate * grad_bias.mean(axis=1)

        return grad_hypothesis

    def concatenate(self, last_layer: Layer):
        if self.input_shape is None:
            self.input_shape = last_layer.output_shape
            self.timesteps, self.input_dim = self.input_shape
            self.output_shape = (self.timesteps, self.units) if self.return_sequence else (self.units,)
            self.truncate_size = self.timesteps if self.truncate_size <= 0 else min(self.truncate_size, self.timesteps)
            # i, f, o, c
            if self.input_dim > 0:
                self.weights = np.random.uniform(size=(4, self.units, self.input_dim))

        assert self.input_shape == last_layer.output_shape, \
            f'The input shape {self.input_shape} of layer "{self.name}" ' \
            f'does not match the output shape {last_layer.output_shape} of its last layer "{last_layer.name}".'
