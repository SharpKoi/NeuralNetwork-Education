from typing import List, Dict, Callable

import math
import numpy as np
from tqdm import tqdm

from sproutnet.nn.layers import Layer
from sproutnet.nn.metric import Metric
from sproutnet.nn.optimizer import Optimizer
from sproutnet.utils import batches


class Model:
    def __init__(self, loss: Callable, optimizer: Optimizer, metric: Metric):
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.layers: List[Layer] = list()
        self.trainable_params: List[np.ndarray] = list()
        self._layers_id: Dict[str, int] = dict()
        self.performance: Dict[str, np.ndarray] = dict()

    def add_layer(self, *layers: Layer):
        for layer in layers:
            if len(self.layers) > 0:
                layer.concatenate(self.layers[-1])
            if layer.name is None:
                ltype = type(layer).__name__
                lid = self._layers_id.get(ltype, 0)
                layer.name = f'{ltype}_{lid}'
                self._layers_id[ltype] = lid + 1
            self.layers.append(layer)
            self.trainable_params.extend(layer.trainable_params)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, batch_size=1, pbar_size=100):
        assert len(X.shape) >= 2 and len(X.shape) >= 2, 'The dim of X and Y cannot be less than 2.'

        self.performance.update({'accuracy': np.zeros(shape=epochs), 'loss': np.zeros(shape=epochs)})
        batched_X, batched_Y = list(batches(X, batch_size)), list(batches(Y, batch_size))
        dataset = list(zip(batched_X, batched_Y))
        for epoch in range(epochs):
            # init progress bar
            _epochs_digits = int(math.log10(epochs))+1
            pbar = tqdm(dataset, ascii=' >=', ncols=pbar_size,
                        desc=f'epoch {str(epoch+1).zfill(_epochs_digits)}/{epochs}: ')

            # training
            accuracy = list()
            loss = list()
            for X_batch, Y_batch in pbar:
                # forward propagation
                output = X_batch
                for layer in self.layers:
                    output = layer.forward_propagate(output)

                # backward propagation
                gradients = list()
                cost = self.loss(output, Y_batch, derivative=True)
                for layer in reversed(self.layers):
                    cost, grads = layer.backward_propagate(cost)
                    gradients = grads + gradients

                self.optimizer.apply_gradients(gradients, self.trainable_params)

                loss.append(self.loss(output, Y_batch))
                if self.metric:
                    accuracy.append(self.metric(output, Y_batch))
                pbar.set_postfix_str(f'accuracy= {np.mean(accuracy):.4f} - loss= {np.mean(loss):.4f}')
                pbar.refresh()

            # update performance
            self.performance['accuracy'][epoch] = np.mean(accuracy)
            self.performance['loss'][epoch] = np.mean(loss)
            pbar.close()

        return self.performance

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
            n_tabs = ((24 - len(layer.name) - int(len(layer.name) % 8 == 0)) // 8)+1
            tabs = '\t' * n_tabs
            msg += f'{layer.name}{tabs}{layer.output_shape}\n'

        return msg
