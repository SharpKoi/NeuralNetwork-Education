from typing import List

import numpy as np

from layers import Layer
from utils import batches


class Model:
    def __init__(self, loss):
        self.loss = loss
        self.layers: List[Layer] = list()

    def add_layer(self, layer: Layer):
        if len(self.layers) > 0:
            layer.concatenate(self.layers[-1])
        if layer.name is None:
            layer.name = f'{type(layer).__name__}_{len(self.layers)}'
        self.layers.append(layer)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float, batch_size=1, verbose=True):
        assert len(X.shape) >= 2 and len(X.shape) >= 2, 'The dim of X and Y cannot be less than 2.'

        batched_X, batched_Y = list(batches(X, batch_size)), list(batches(Y, batch_size))
        dataset = list(zip(batched_X, batched_Y))
        for epoch in range(epochs):
            error = 0
            for X_batch, Y_batch in dataset:
                # forward propagation
                output = X_batch
                for layer in self.layers:
                    output = layer.forward_propagate(output)

                # backward propagation
                cost = self.loss(output, Y_batch, derivative=True)
                for layer in reversed(self.layers):
                    cost = layer.backward_propagate(cost, learning_rate)

                error += self.loss(output, Y_batch).sum(axis=0)

            error /= len(Y)
            if verbose:
                print(f'epoch {epoch}:\tloss= {error}')

    def predict(self, X: np.ndarray):
        output = X.reshape(1, -1) if len(X.shape) == 1 else X
        for layer in self.layers:
            output = layer.forward_propagate(output)

        return output[0] if len(X.shape) == 1 else output

    def __repr__(self):
        return 'Layers\t\tOutput Shape\n' + \
               '_____________________________\n' + \
               '\n'.join([f'{layer.name}:\t\t{layer.output_shape}' for layer in self.layers])
