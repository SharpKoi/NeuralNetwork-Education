import numpy as np


def softmax(x, derivative=False) -> np.ndarray:
    # more stable softmax. Avoid inf. value
    x_exp = np.exp(x-np.max(x))
    y = x_exp / np.sum(x_exp)

    if derivative:
        n = len(x)
        y = np.array([[y[i] * (int(i == j) - y[j]) for j in range(n)] for i in range(n)])

    return y


def tanh(x, derivative=False):
    y = np.tanh(x)
    if derivative:
        return 1-np.square(y)

    return y


def sigmoid(x, derivative=False):
    x_limit = int(np.log(np.finfo('d').max))
    ool = x[np.abs(x) > x_limit]
    x[np.abs(x) > x_limit] = np.sign(ool) * x_limit
    y = 1 / (1 + np.exp(-x))
    if derivative:
        return y * (1 - y)

    return y


def ReLU(x, derivative=False):
    y = np.maximum(x, 0)
    if derivative:
        return (y > 0).astype(float)

    return y
