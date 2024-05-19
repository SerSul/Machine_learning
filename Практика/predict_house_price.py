import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:

    def __init__(self, learning_rate=0.0000000001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None

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


house_sizes = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)

house_prices = np.array([200000, 300000, 400000, 500000, 600000])


model = LinearRegressionGD(learning_rate=0.00001, n_iter=10000)
model.fit(house_sizes, house_prices)


test_sizes = np.array([[1000], [1100], [1200], [1300], [1400], [1500], [1600], [1700], [1800], [1900], [2000], [2100], [2200], [2300], [2400], [2500], [2600], [2700], [2800], [2900]])
predicted_prices = model.predict(test_sizes)

for i in predicted_prices:
    print(i)


plt.figure(figsize=(10, 6))
plt.scatter(house_sizes, house_prices, color='blue', label='Изначальные данные')
plt.plot(test_sizes, predicted_prices, color='red', label='Предсказанные цены')
plt.xlabel('Площадь дома (кв. фт)')
plt.ylabel('Цена дома ($)')
plt.title('Линейная регрессия: предсказание цен на дома')
plt.legend()
plt.grid(True)
plt.show()