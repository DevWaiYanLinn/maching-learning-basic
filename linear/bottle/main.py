import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

bottle_df = pd.read_csv("194903-202105_Bottle.csv", encoding='unicode_escape')
bottle_df.info()
bottle_df_cust = bottle_df[['Salnty', 'Depthm', 'T_degC']].copy()
bottle_df_cust.head()
bottle_df_cust.isnull().sum()

from sklearn.impute import SimpleImputer

si = SimpleImputer()
bottle_df_cust = pd.DataFrame(si.fit_transform(bottle_df_cust), columns=['Salnty', 'Depthm', 'T_degC'])
bottle_df_cust.isnull().sum()
import seaborn as sns

sns.pairplot(bottle_df_cust, x_vars=['Salnty', 'Depthm'], y_vars=['T_degC'])
# plt.show()
X = bottle_df_cust[['Salnty', 'Depthm']].copy()
Y = bottle_df_cust['T_degC'].copy()

print(X.shape, Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=39)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn import linear_model

lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)

print("Coefficients of the model:", lm.coef_)
print("Intercept of the mode:", lm.intercept_)

y_pred = lm.predict(X_test)

from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean(y_pred - y_test) ** 2)
print("R2-score: %.2f" % r2_score(y_pred, y_test))

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_train_poly = poly_reg.fit_transform(X_train)
lm.fit(X_train_poly, y_train)
X_test_poly = poly_reg.fit_transform(X_test)
y_pred = lm.predict(X_test_poly)

from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_pred, y_test))
