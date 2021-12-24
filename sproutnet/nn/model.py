from typing import List, Callable

import numpy as np
from tqdm import tqdm

from sproutnet.nn.layers import Layer
from sproutnet.nn.metric import Metric
from sproutnet.utils import batches


class Model:
    def __init__(self, loss: Callable, metric: Metric):
        self.loss = loss
        self.metric = metric
        self.layers: List[Layer] = list()

    def add_layer(self, *layers: Layer):
        layers_id = dict()
        for layer in layers:
            if len(self.layers) > 0:
                layer.concatenate(self.layers[-1])
            if layer.name is None:
                ltype = type(layer).__name__
                lid = layers_id.get(ltype, 0)
                layer.name = f'{ltype}_{lid}'
                layers_id[ltype] = lid + 1
            self.layers.append(layer)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int,
            learning_rate: float, batch_size=1,
            pbar_size=100):
        assert len(X.shape) >= 2 and len(X.shape) >= 2, 'The dim of X and Y cannot be less than 2.'

        batched_X, batched_Y = list(batches(X, batch_size)), list(batches(Y, batch_size))
        dataset = list(zip(batched_X, batched_Y))
        for epoch in range(epochs):
            accuracy = list()
            loss = list()
            pbar = tqdm(dataset, ascii=' >=', desc=f'epoch {epoch}: ', ncols=pbar_size)
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
                if self.metric:
                    accuracy.append(self.metric(output, Y_batch))
                pbar.set_postfix({'accuracy': np.mean(accuracy), 'loss': np.mean(loss)})
                pbar.refresh()
            pbar.close()

    def evaluate(self, X: np.ndarray, Y: np.ndarray):
        y_pred = self.predict(X)
        return {'accuracy': self.metric(y_pred, Y), 'loss': self.loss(y_pred, Y)}

    def predict(self, X: np.ndarray):
        output = X.reshape(1, -1) if len(X.shape) == 1 else X
        for layer in self.layers:
            output = layer.forward_propagate(output)

        return output[0] if len(X.shape) == 1 else output

    def __repr__(self):
        msg = 'Layers\t\t\tOutput Shape\n' + \
              '____________________________________\n'
        for layer in self.layers:
            prefix_size = len(layer.name) + 1
            n_tabs = ((20 - prefix_size - int(prefix_size % 4 == 0)) // 4)
            tabs = '\t' * n_tabs
            msg += f'{layer.name}:{tabs}{layer.output_shape}\n'

        return msg
