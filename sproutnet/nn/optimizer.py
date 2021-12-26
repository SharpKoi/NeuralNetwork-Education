from abc import ABC

import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate, name: str):
        self.learning_rate = learning_rate
        self.name = name

    def apply_gradients(self, gradients, trainable_params):
        raise NotImplementedError


class BGD(Optimizer):
    def __init__(self, learning_rate=1e-2, name='bgd'):
        super(BGD, self).__init__(learning_rate, name)

    def apply_gradients(self, gradients, trainable_params):
        for i in range(len(trainable_params)):
            trainable_params[i] -= self.learning_rate * gradients[i]


class Adam(Optimizer):
    def __init__(self,
                 learning_rate=1e-3,
                 decay_rate_1=0.9,
                 decay_rate_2=0.999,
                 epsilon=1e-8,
                 name='adam'):
        super(Adam, self).__init__(learning_rate, name)
        self.decay_rate = (decay_rate_1, decay_rate_2)
        self.latest_moments = None
        self.epsilon = epsilon

    def apply_gradients(self, gradients, trainable_params):
        # init latest moments
        if self.latest_moments is None:
            self.latest_moments = [[np.zeros_like(grad), np.zeros_like(grad)] for grad in gradients]

        for i in range(len(gradients)):
            b1, b2 = self.decay_rate
            self.latest_moments[i][0] = b1 * self.latest_moments[i][0] + (1 - b1) * gradients[i]
            self.latest_moments[i][1] = b2 * self.latest_moments[i][1] + (1 - b2) * gradients[i]**2
            m = self.latest_moments[i][0] / (1 - b1)
            v = self.latest_moments[i][1] / (1 - b2)

            trainable_params[i] -= self.learning_rate * m / (np.sqrt(v) + self.epsilon)
