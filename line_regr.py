import numpy as np


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0):
        self.n_iter = n_iter
        if callable(learning_rate):
            self.learning_rate_func = learning_rate
        else:
            self.learning_rate_func = lambda _: learning_rate
        self.weights = weights
        if metric is None or metric in ['mae', 'mse', 'rmse', 'mape', 'r2']:
            self.metric = metric
        else:
            self.metric = None
        self.final_metric = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, metric={self.metric}"

    """ Обучение модели """

    def fit(self, X, y, verbose=False):
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.ones(X.shape[1]) if self.weights is None else self.weights
        y_pred = X.dot(self.weights)
        initial_loss = self._calculate_loss(y, y_pred)
        if verbose:
            print(f"start | loss: {initial_loss:.2f} | {self.metric}: {self._calculate_metric(y, y_pred):.2f}")
        for i in range(self.n_iter):
            y_pred = X.dot(self.weights)
            errors = y - y_pred
            gradient = -2 * X.T.dot(errors) / X.shape[0]

            gradient += self._calculate_grad()
            learning_rate = self.learning_rate_func(i + 1)
            self.weights -= learning_rate * gradient

            if verbose and (i + 1) % verbose == 0:
                loss = self._calculate_loss(y, y_pred)
                metric_value = self._calculate_metric(y, y_pred)
                print(f"{i + 1} | loss: {loss:.2f} | {self.metric}: {metric_value:.2f}")

        y_pred = X.dot(self.weights)
        self.final_metric = self._calculate_metric(y, y_pred)

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.weights)

    def get_coef(self):
        return self.weights[1:]

    def _calculate_loss(self, y, y_pred):
        base_loss = np.mean((y - y_pred) ** 2)
        reg_loss = self._calculate_reg()
        return base_loss + reg_loss

    def _calculate_metric(self, y, y_pred):
        if self.metric == 'mae': # Mean Absolute Error - средняя абсолютная ошибка
            return np.mean(np.abs(y - y_pred))
        elif self.metric == 'mse': # Mean Squared Error - средняя квадратичная ошибка
            return np.mean((y - y_pred) ** 2)
        elif self.metric == 'rmse': # Root Mean Squared Error - корень из средней квадратичной ошибки
            return np.sqrt(np.mean((y - y_pred) ** 2))
        elif self.metric == 'mape': # Mean Absolute Percentage Error - средняя абсолютная процентная ошибка
            return np.mean(np.abs((y - y_pred) / y)) * 100
        elif self.metric == 'r2': # R^2 - коэффициент детерминации
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)

    def get_best_score(self):
        return self.final_metric

    def _calculate_reg(self):
        if self.reg == 'l1':
            return self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == 'l2':
            return self.l2_coef * np.sum(self.weights ** 2)
        elif self.reg == 'elasticnet':
            return self.l1_coef * np.sum(np.abs(self.weights[1:])) + self.l2_coef * np.sum(self.weights ** 2)
        else:
            return 0

    def _calculate_grad(self):
        if self.reg == 'l1':
            return self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2':
            return 2 * self.l2_coef * self.weights
        elif self.reg == 'elasticnet':
            return self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
        else:
            return 0


features = np.array([
    [12, 42],
    [24, 33],
])

target = np.array([3, 2])

model = MyLineReg(n_iter=1000000, learning_rate=0.1, metric='mape', reg='elasticnet', l1_coef=0.5, l2_coef=0.5)
model.fit(features, target, verbose=1000)
print(model.predict(features))
