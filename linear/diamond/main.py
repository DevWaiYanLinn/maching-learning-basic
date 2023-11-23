import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv('diamonds.csv')

data.shape
data.info()
data.describe()
data.head()

missing_values = data.isnull().sum()
print(missing_values)

data = data.fillna(data.mean(numeric_only=True))
duplicate_values = data[data.duplicated()]
print(duplicate_values)

data = data.drop_duplicates()

data = data.drop('Unnamed: 0', axis=1)

Q1 = data.quantile(0.25, numeric_only=True)
Q3 = data.quantile(0.75, numeric_only=True)

IQR = Q3 - Q1
# data = data[((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]
X = data['carat']
y = data['price']

# data.shape


sns.pairplot(data, x_vars=['carat'], y_vars=['price'], height=12, kind='scatter')
plt.xlabel('Carat')
plt.xlabel('Price')
plt.title('Diamond Price Prediction - Carat vs Price')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

X_train_array = np.array(X_train)
X_train = X_train_array[:, np.newaxis]

X_test_array = np.array(X_test)
X_test = X_test_array[:, np.newaxis]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_scaled, y_train)

y_pred = linear_regression_model.predict(X_test_scaled)
plt.figure(figsize=(12, 6))
idx = [i for i in range(1, len(y_test) + 1, 1)]
plt.plot(idx, y_test, color='r', linestyle='-')
plt.plot(idx, y_pred, color='b', linestyle='-')

plt.xlabel('Price')
plt.ylabel('Index')
plt.title('Prediction')
plt.show()

from sklearn.metrics import r2_score, mean_squared_error

print("Mean Squared Error", mean_squared_error(y_test, y_pred))
print("R2 Score", r2_score(y_test, y_pred))

print('Intercept of the model:', linear_regression_model.intercept_)
print('Cofficient of the line:', linear_regression_model.coef_)

feature_to_visualize = 'carat'
scaled_feature = scaler.transform(data[[feature_to_visualize]])

predicted_prices = linear_regression_model.predict(scaled_feature)
residuals = data['price'] - predicted_prices

plt.figure(figsize=(12, 6))
plt.scatter(predicted_prices, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')

plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

feature_to_visualize = "carat"

plt.figure(figsize=(12, 6))

plt.scatter(data[feature_to_visualize], data["price"], alpha=0.3, label="Actual Value")
plt.plot(data[feature_to_visualize], predicted_prices, color="red", linewidth=3, label="Regression Line")

plt.xlabel(feature_to_visualize)
plt.ylabel("Price")
plt.title('Scatter plot with Regression line')

plt.legend()
plt.show()

import pickle

with open('linear_regression_model.pkl', "wb") as model_file:
    try:
        pickle.dump(linear_regression_model, model_file)
        print("Model save successfully.")
    except Exception as error_msg:
        print("Error: Model saving failed", error_msg)


use_trained_model = str(input("DO you want tou sue the trained model instead"))

if use_trained_model == 'Y' or 'y':
    model = linear_regression_model
    print("Trained model loaded successfully")
else:
    print(error_msg)

input_carat = float(input('Enter the carats: '))

if model:
    predicted_prices = model.predict([[input_carat]])[0]
    print(f'Predicted Price for a {input_carat} carat diamond is ${predicted_prices}')
else:
    print("Model not found. Try loading the save model again or infer from the trained model")
