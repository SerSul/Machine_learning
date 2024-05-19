import numpy as np

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

        self.coefficients = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T.dot(y)) # МНК

    def predict(self, X):
        """
        Предсказание
        @:param X: признаки
        @:return: предсказанные значения
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b.dot(self.coefficients)
