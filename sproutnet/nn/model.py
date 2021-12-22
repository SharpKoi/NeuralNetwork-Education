from typing import List, Callable

import numpy as np
from tqdm import tqdm

from sproutnet.nn.layers import Layer
from sproutnet.nn.metric import Metric
from sproutnet.utils import batches


class Model:
    def __init__(self, loss: Callable):
        self.loss = loss
        self.layers: List[Layer] = list()

    def add_layer(self, *layers: Layer):
        for layer in layers:
            if len(self.layers) > 0:
                layer.concatenate(self.layers[-1])
            if layer.name is None:
                layer.name = f'{type(layer).__name__}_{len(self.layers)}'
            self.layers.append(layer)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float, batch_size=1, metric: Metric = None):
        assert len(X.shape) >= 2 and len(X.shape) >= 2, 'The dim of X and Y cannot be less than 2.'

        batched_X, batched_Y = list(batches(X, batch_size)), list(batches(Y, batch_size))
        dataset = list(zip(batched_X, batched_Y))
        for epoch in range(epochs):
            accuracy = list()
            loss = list()
            pbar = tqdm(dataset, ascii=True, desc=f'epoch {epoch}: ')
            for X_batch, Y_batch in pbar:
                # forward propagation
                output = X_batch
                for layer in self.layers:
                    output = layer.forward_propagate(output)

                # backward propagation
                cost = self.loss(output, Y_batch, derivative=True)
                for layer in reversed(self.layers):
                    cost = layer.backward_propagate(cost, learning_rate)

                loss.append(self.loss(output, Y_batch))
                if metric:
                    accuracy.append(metric(Y_batch, output))
                pbar.set_postfix({'accuracy': np.mean(accuracy), 'loss': np.mean(loss)})
                pbar.refresh()

    def predict(self, X: np.ndarray):
        output = X.reshape(1, -1) if len(X.shape) == 1 else X
        for layer in self.layers:
            output = layer.forward_propagate(output)

        return output[0] if len(X.shape) == 1 else output

    def __repr__(self):
        msg = 'Layers\t\t\tOutput Shape\n' + \
              '_____________________________\n'
        for layer in self.layers:
            n_tabs = ((20 - len(layer.name) - 1) // 4)
            if n_tabs < 2:
                n_tabs += 1
            tabs = '\t' * n_tabs
            msg += f'{layer.name}:{tabs}{layer.output_shape}\n'

        return msg
