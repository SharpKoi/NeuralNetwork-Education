from typing import Tuple, Callable, Optional, List

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
        the input shape without batch size thus should be 2d (sequence length, input dim)
    activator: Callable
        the activator function used to activate i, f, o gates in each cell
    cell_activator: Callable
        the activator function used to activate c gate and hypothesis in each cell
    weights_initializer: Initializer
        the initializer of input weights. default is `sproutnet.nn.initializer.GlorotUniform`
    state_weights_initializer: Initializer
        the initializer of state weights. default is `sproutnet.nn.initializer.Orthogonal`
    bias_initializer: Initializer
        the initializer of bias. default is `sproutnet.nn.initializer.Zeros`
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
        self.trainable_params = [None] * 3
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
                self.trainable_params[0] = self.weights

        self.state_weights = self.state_weights_initializer(shape=(4, self.units, self.units))
        self.trainable_params[1] = self.state_weights
        self.bias = self.bias_initializer(shape=(4, self.units))
        self.trainable_params[2] = self.bias

        # declare gates and signals
        self.raw_gates = None
        self.cell_states = None
        self.hidden_states = None

    def forward_propagate(self, input_data: np.ndarray) -> np.ndarray:
        # shape of input_data: (batch_size, timesteps, input_dim)
        batch_size, n_timesteps, n_features = input_data.shape

        assert n_timesteps == self.timesteps and n_features == self.input_dim

        self.input = input_data

        self.raw_gates = np.zeros(shape=(4, batch_size, self.timesteps, self.units))
        gates = np.empty(shape=(4, batch_size, self.timesteps, self.units))

        self.cell_states = np.zeros(shape=(batch_size, self.timesteps+1, self.units))
        self.hidden_states = np.copy(self.cell_states)

        for i in range(batch_size):
            for t in range(self.timesteps):
                # get x_t, h_t-1, c_t-1
                x = input_data[i, t, :]
                h = self.hidden_states[i, t, :]
                c = self.cell_states[i, t, :]

                # ====================== start computing feedforward ====================== #
                for j in range(4):
                    self.raw_gates[j, i, t, :] = np.dot(self.weights[j], x) + np.dot(self.state_weights[j], h) + self.bias[j]

                gates[:-1, i, t, :] = self.activator(self.raw_gates[:-1, i, t, :])
                gates[-1, i, t, :] = self.cell_activator(self.raw_gates[-1, i, t, :])

                gate_i, gate_f, gate_o, gate_c = gates
                self.cell_states[i, t+1, :] = gate_f[i, t, :] * c + gate_i[i, t, :] * gate_c[i, t, :]
                self.hidden_states[i, t+1, :] = gate_o[i, t, :] * self.cell_activator(self.cell_states[i, t+1, :])
                # ====================== end of feedforward ====================== #

        if self.return_sequence:
            self.output = self.hidden_states[:, 1:, :]
        else:
            self.output = self.hidden_states[:, -1, :]

        return self.output

    def backward_propagate(self, output_error: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        # output_error (batch_size, units)
        batch_size = output_error.shape[0]

        grad_gates = np.zeros(shape=(4, batch_size, self.units))
        # grad_cell_state = np.zeros(shape=(batch_size, self.timesteps+1, self.units))
        grad_weights = np.zeros(shape=(4, batch_size, *self.weights.shape[1:]))
        grad_state_weights = np.zeros(shape=(4, batch_size, *self.state_weights.shape[1:]))
        grad_bias = np.zeros(shape=(4, batch_size, *self.bias.shape[1:]))
        grad_hypothesis = np.zeros(shape=self.input.shape)

        truncated = False
        # ====================== starting gradient computation ====================== #
        for t in reversed(range(self.timesteps)):
            raw_i, raw_f, raw_o, raw_c = self.raw_gates

            grad_cell_state = output_error * \
                                       self.activator(raw_o[:, t, :]) * \
                                       self.cell_activator(self.cell_states[:, t+1, :], derivative=True)
            # i
            grad_gates[0] = grad_cell_state * \
                            self.cell_activator(raw_c[:, t, :]) * \
                            self.activator(raw_i[:, t, :], derivative=True)
            # f
            grad_gates[1] = grad_cell_state * \
                            self.cell_states[:, t, :] * \
                            self.activator(raw_f[:, t, :], derivative=True)
            # o
            grad_gates[2] = output_error * \
                            self.cell_activator(self.cell_states[:, t+1, :]) * \
                            self.activator(raw_o[:, t, :], derivative=True)
            # c
            grad_gates[3] = grad_cell_state * \
                            self.activator(raw_i[:, t, :]) * \
                            self.cell_activator(raw_c[:, t, :], derivative=True)

            # totally 4 gates: i, f, o, c
            for j in range(4):
                # compute the next output error
                output_error += np.matmul(grad_gates[j], self.state_weights[j])

                # update gradients only if not truncated
                if not truncated:
                    for i in range(batch_size):
                        grad_weights[j, i] += np.outer(grad_gates[j, i, :], self.input[i, t, :])
                        grad_state_weights[j, i] += np.outer(grad_gates[j, i, :], self.hidden_states[i, t, :])
                    grad_bias[j] += grad_gates[j]

                # (batch_size, input_dim) <- (batch_size, units)(units, input_dim)
                grad_hypothesis[:, t, :] += np.matmul(grad_gates[j], self.weights[j])

            truncated = ((self.timesteps - t) % self.truncate_size) == 0
        # ====================== end of gradient computation ====================== #

        # do BPTT
        grad_params = list()
        if self.trainable:
            grad_params = [grad_weights.mean(axis=1), grad_state_weights.mean(axis=1), grad_bias.mean(axis=1)]
            # self.weights -= learning_rate * grad_weights.mean(axis=1)
            # self.state_weights -= learning_rate * grad_state_weights.mean(axis=1)
            # self.bias -= learning_rate * grad_bias.mean(axis=1)

        return grad_hypothesis, grad_params

    def concatenate(self, last_layer: Layer):
        if self.input_shape is None:
            self.input_shape = last_layer.output_shape
            self.timesteps, self.input_dim = self.input_shape
            self.output_shape = (self.timesteps, self.units) if self.return_sequence else (self.units,)
            self.truncate_size = self.timesteps if self.truncate_size <= 0 else min(self.truncate_size, self.timesteps)
            # i, f, o, c
            if self.input_dim > 0:
                self.weights = np.random.uniform(size=(4, self.units, self.input_dim))
                self.trainable_params[0] = self.weights

        assert self.input_shape == last_layer.output_shape, \
            f'The input shape {self.input_shape} of layer "{self.name}" ' \
            f'does not match the output shape {last_layer.output_shape} of its last layer "{last_layer.name}".'
