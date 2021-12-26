import numpy as np


def mean_squared_error(output: np.ndarray, target: np.ndarray, derivative=False) -> np.ndarray:
    assert output.shape == target.shape

    if derivative:
        return np.mean(2 * (output - target), axis=-1)
    else:
        # reduction by mean
        return np.mean(np.mean(np.square(target - output), axis=-1))


def binary_cross_entropy(output: np.array, target: np.array, epsilon=1e-12, derivative=False) -> np.ndarray:
    assert output.shape == target.shape

    output = np.clip(output, epsilon, 1-epsilon)
    if derivative:
        return (output-target) / np.maximum(output*(1-output), epsilon)
    else:
        return -np.sum(np.mean(target*np.log(output) + (1-target)*np.log(1-output), axis=0), axis=-1)


# def categorical_cross_entropy()
