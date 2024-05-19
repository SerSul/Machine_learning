import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


class LinearRegression:
    """ Линейная регрессия методом наименьших квадратов"""
    def __init__(self, coefficients=None):
        self.coefficients = coefficients

    def fit(self, X, y):
        """
        Обучение модели методом наименьших квадратов
        @:param X: признаки
        @:param y: целевая переменная
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        X_transpose_X = X_b.T.dot(X_b)

        X_transpose_X_inv = np.linalg.inv(X_transpose_X)

        X_transpose_y = X_b.T.dot(y)

        self.coefficients = X_transpose_X_inv.dot(X_transpose_y)

    def predict(self, X):
        """
        Предсказание
        @:param X: признаки
        @:return: предсказанные значения
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b.dot(self.coefficients)

