import numpy as np


def mean_squared_error(output: np.ndarray, target: np.ndarray, derivative=False) -> np.ndarray:
    assert output.shape == target.shape

    n_labels = output.shape[-1]
    if derivative:
        # return 2 * (output - target) / n_labels
        return output - target
    else:
        return np.mean(np.square(target - output), axis=-1)


def binary_cross_entropy(output: np.array, target: np.array, epsilon=1e-8, derivative=False) -> np.ndarray:
    output = np.clip(output, epsilon, 1-epsilon)
    if derivative:
        return (output-target) / np.maximum(output*(1-output), epsilon)
    else:
        return -np.mean(np.sum(target*np.log(output) + (1-target)*np.log(1-output), axis=-1))


# def categorical_cross_entropy()
