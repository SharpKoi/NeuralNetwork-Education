import numpy as np


def softmax(x, derivative=False):
    x_exp = np.exp(x-np.max(x))     # avoid Inf values
    y = x_exp / x_exp.sum()
    if derivative:
        # TODO: find the derivation of softmax function.

        pass

    return y


def tanh(x, derivative=False):
    y = np.tanh(x)
    if derivative:
        return 1-np.square(x)

    return y


def sigmoid(x, derivative=False):
    y = 1 / (1 + np.exp(-x))
    if derivative:
        return y * (1 - y)

    return y


def ReLU(x, derivative=False):
    y = np.maximum(x, 0)
    if derivative:
        return (y > 0).astype(float)

    return y
