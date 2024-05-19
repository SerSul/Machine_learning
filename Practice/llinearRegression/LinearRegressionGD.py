import numpy as np


class LinearRegressionGD:
    """ Линейная регрессия методом градиентного спуска"""
    def __init__(self, learning_rate=0.01, n_iters=1000, weights=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = weights

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape

        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        X = np.c_[np.ones(n_samples), X]

        if self.weights is None:
            self.weights = np.zeros(n_features + 1)

        for i in range(self.n_iters):
            predicted_y = np.dot(X, self.weights)
            loss = np.mean((predicted_y - y) ** 2)
            if verbose and (i + 1) % verbose == 0:
                print(f"{i + 1} | loss: {loss:.3f}")
            gradient = np.dot(X.T, (predicted_y - y)) / n_samples
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        n_samples = X.shape[0]

        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        X = np.c_[np.ones(n_samples), X]

        return np.dot(X, self.weights)

