import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __repr__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        X = np.array(X)
        y = np.array(y)

        X = np.c_[np.ones(X.shape[0]), X]

        self.weights = np.ones(X.shape[1])

        if verbose:
            y_pred = self.pred(X)
            start_loss = self.compute_log_loss(y, y_pred)
            print(f"start | loss: {start_loss:.2f}")

        for i in range(self.n_iter):
            y_pred = self.pred(X)
            log_loss = self.compute_log_loss(y, y_pred)
            gradient = np.dot(X.T, (y_pred - y)) / X.shape[0]
            self.weights -= self.learning_rate * gradient
            if verbose and (i + 1) % verbose == 0:
                print(f"{i + 1} | loss: {log_loss:.2f}")

    def pred(self, X):
        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return self.pred(X) > 0.5

    def predict_proba(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return self.pred(X)

    def sigmoid(self, z):
        """ Сигмоидная функция """
        return 1 / (1 + np.exp(-z))

    def compute_log_loss(self, y_true, y_pred):
        """ Логистическая функция потерь """
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)

    def get_coef(self):
        return np.array(self.weights[1:])



X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_custom = MyLogReg(n_iter=500, learning_rate=0.1)
model_custom.fit(X_train, y_train, verbose=True)
y_pred_custom = model_custom.predict_proba(X_test)

model_sklearn = LogisticRegression(max_iter=500, solver='lbfgs')
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict_proba(X_test)


print("Custom Model Log Loss:", model_custom.compute_log_loss(y_test, y_pred_custom))
print("Sklearn Model Log Loss:", log_loss(y_test, y_pred_sklearn))

print("Custom Model Coefficients:", model_custom.get_coef())

print("Sklearn Model Coefficients:", model_sklearn.coef_)