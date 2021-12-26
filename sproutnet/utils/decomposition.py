import numpy as np


class PCA:
    def __init__(self, n_components: int, seed=None):
        self.n_components = n_components
        self.seed = seed

    def fit_transform(self, data):
        # (n_samples, n_dim)
        data -= data.mean(axis=0)
        cov = np.cov(data, rowvar=False)
        eigen_vals, eigen_vecs = self._get_eigens(cov)
        result = np.matmul(data, eigen_vecs.T)

        return result

    def _significant_eigen(self, matrix: np.ndarray, tol=1e-8):
        assert matrix.shape[0] == matrix.shape[1], 'The matrix is not a square matrix.'

        np.random.seed(self.seed)
        eig_vec = np.random.rand(matrix.shape[1])
        while True:
            z = np.dot(matrix, eig_vec)
            new_eig_vec = z / np.linalg.norm(z) if z.any() else np.zeros_like(eig_vec)

            loss = np.linalg.norm(np.abs(new_eig_vec) - np.abs(eig_vec))
            if loss <= tol:
                break

            eig_vec = new_eig_vec

        # Ax = ux => u = Ax/x
        eig_val = (np.dot(matrix, new_eig_vec) / new_eig_vec).mean() if new_eig_vec.any() else 0

        return eig_val, new_eig_vec

    def _get_eigens(self, matrix: np.ndarray):
        dim = matrix.shape[0]
        n_eigens = min(dim, self.n_components)
        eigen_vals = np.empty(shape=(n_eigens,))
        eigen_vecs = np.empty(shape=(n_eigens, dim))
        for i in range(n_eigens):
            eig_val, eig_vec = self._significant_eigen(matrix)
            eigen_vals[i] = eig_val
            eigen_vecs[i] = eig_vec

            # Apply Wielandt's deflation
            eig_vec = eig_vec.reshape(-1, 1)
            # Note that eig_vec is normalized, we can choose eig_vec as v s.t. (v.T)(eig_vec) = 1
            matrix -= eig_val * np.matmul(eig_vec, eig_vec.T)

        return eigen_vals, eigen_vecs
