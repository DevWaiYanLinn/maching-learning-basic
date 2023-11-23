import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import sys

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def plot_regression_line(X, y, model, title):
    plt.scatter(X, y, alpha=0.5, label='Data')
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    print(x_range)
    sys.exit()
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, label='Regression Line')
    plt.title(title)
    plt.legend()
    plt.show()


def evalute_model(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{title} - Mean Squared Error on Test Set: {mse}')


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
plot_regression_line(X, y, linear_model, 'Linear Regression (Good Fit)')
evalute_model(linear_model, X_test, y_test, 'Linear Regression (Good Fit)')

underfit_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()) #bx**3 + bx + a
underfit_model.fit(X_train, y_train)
plot_regression_line(X, y, underfit_model, 'Underfitting (linear Model for Quadratic Data)')
evalute_model(underfit_model, X_test, y_test, 'Underfitting')

overfit_model = make_pipeline(PolynomialFeatures(degree=15), LinearRegression())
overfit_model.fit(X_train, y_train)
plot_regression_line(X, y, overfit_model, 'Overfitting (Hig-Degree Polynomial)')
evalute_model(overfit_model, X_test, y_test, 'Overfitting')
