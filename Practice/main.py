import numpy as np
from llinearRegression import LinearRegression, LinearRegressionSG
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class TestLinearRegression:

    def __init__(self):
        df = pd.read_csv('house-prices.csv')
        self.house_prices = df.iloc[:, 1].values
        self.house_sizes = df.iloc[:, 2].values

        self.train_sizes, self.test_sizes, self.train_prices, self.test_prices = train_test_split(
            self.house_sizes, self.house_prices, test_size=0.2, random_state=42)

        print("Linear Regression")
        self.test_linear_regression()

        print("\nSGD")
        self.test_linear_regression_sg()

    def test_linear_regression(self):
        model = LinearRegression.LinearRegression()
        model.fit(self.train_sizes.reshape(-1, 1), self.train_prices)

        self.predicted_prices = model.predict(self.test_sizes.reshape(-1, 1))

    def test_linear_regression_sg(self):
        model = LinearRegressionSG.LinearRegressionGD(learning_rate=0.1, n_iters=10000)
        model.fit(self.train_sizes.reshape(-1, 1), self.train_prices)

        self.predicted_prices_sg = model.predict(self.test_sizes.reshape(-1, 1))

    def print_diff(self):
        print("\nDifference between Linear Regression and GD:")
        for real_price, predicted_price, predicted_price_sg in zip(self.test_prices, self.predicted_prices,
                                                                   self.predicted_prices_sg):
            print(
                f"Real: {real_price:.3f} | Predicted (LR): {predicted_price:.3f} | Predicted (GD): {predicted_price_sg:.3f}")


if __name__ == '__main__':
    tester = TestLinearRegression()
    tester.print_diff()
