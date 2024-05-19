import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from llinearRegression import LinearRegression, LinearRegressionGD


class TestLinearRegression:

    def __init__(self):
        df = pd.read_csv('house-prices.csv')
        self.house_prices = df.iloc[:, 1].values
        self.house_sizes = df.iloc[:, 2].values
        self.bedrooms = df.iloc[:, 3].values
        self.bathrooms = df.iloc[:, 4].values
        self.neighborhood = df.iloc[:, 7].values

        self.neighborhood_encoder = OneHotEncoder(sparse_output=False)
        self.neighborhood_encoded = self.neighborhood_encoder.fit_transform(self.neighborhood.reshape(-1, 1))

        X = pd.DataFrame({
            'sizes': self.house_sizes,
            'bedrooms': self.bedrooms,
            'bathrooms': self.bathrooms,
            **pd.DataFrame(self.neighborhood_encoded,
                           columns=self.neighborhood_encoder.get_feature_names_out(['neighborhood']))
        })
        y = self.house_prices
        self.train_sizes, self.test_sizes, self.train_prices, self.test_prices = train_test_split(
            X, y, test_size=0.2, random_state=42)
        print("Linear Regression")
        self.test_linear_regression()

        print("\nSGD")
        self.test_linear_regression_sg()

    def test_linear_regression(self):
        model = LinearRegression.LinearRegression()
        model.fit(self.train_sizes, self.train_prices)
        self.predicted_prices = model.predict(self.test_sizes)

    def test_linear_regression_sg(self):
        model = LinearRegressionGD.LinearRegressionGD()
        model.fit(self.train_sizes, self.train_prices)
        self.predicted_prices_sg = model.predict(self.test_sizes)

    def print_diff(self):
        print("\nDifference between Linear Regression and SGD:")
        for real_price, predicted_price, predicted_price_sg, size, bedroom, bathroom, neighborhood in zip(
                self.test_prices, self.predicted_prices, self.predicted_prices_sg,
                self.test_sizes['sizes'], self.test_sizes['bedrooms'],
                self.test_sizes['bathrooms'], self.neighborhood_encoder.inverse_transform(self.test_sizes.iloc[:, 3:])):
            print(
                f"Size: {size:.3f} | Bedrooms: {bedroom:.3f} | Bathrooms: {bathroom:.3f} | Neighborhood: {neighborhood[0]}")
            print(
                f"Real: {real_price:.3f} | Predicted (LR): {predicted_price:.3f} | Predicted (SGD): {predicted_price_sg:.3f}")
            print()


if __name__ == '__main__':
    tester = TestLinearRegression()
    tester.print_diff()
