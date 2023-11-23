import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import  matplotlib.cm as cm

df = pd.read_csv('./BostonHousing.csv')
df = df[df['zn'] > 0]
X = df[["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]].values
y = df['medv'].values

corrMatrix = df.corr()
sns.heatmap(corrMatrix,annot=True, cmap=cm.RdYlGn)
plt.show()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=144++)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_predict = model.predict(X_test)
# msc = mean_squared_error(y_test, y_predict)
# print(f'Mean square error {msc}')
# plt.scatter(y_test, y_predict)
# plt.plot(y_test, y_test)
# plt.show()
