import sys

from sklearn.datasets import make_blobs, make_moons, make_circles
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# X, y = make_blobs(n_samples=100, centers=3, cluster_std=1, n_features=3)
#
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
# ax.set_xlabel('Feature1')
# ax.set_ylabel('Feature2')
# ax.set_zlabel('Feature3')
# ax.legend()
# plt.show()

# np.random.seed(0)
#
# X = np.random.rand(100, 1)
#
# y = 2 * X + 1 + 0.1 * np.random.rand(100, 1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 80-20
#
# model = LinearRegression()
#
# model.fit(X_train, y_train)
#
# y_predict = model.predict(X_test)
# print(model.predict([[2]]))
# mse = mean_squared_error(y_test, y_predict)  # model accuracy
# print(f'Mean squared error: {mse}')
#
# plt.scatter(X_test, y_test, color='blue')
# plt.plot(X_test, y_predict, color='red')
# plt.x


np.random.seed(1)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 80-20
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
acs = accuracy_score(y_test, y_predict)  # model accuracy
print(f'Accuracy Score: {acs}')

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='Dark2')

ax = plt.gca()
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
# print(x_lim[0])
# print(np.linspace(x_lim[0], x_lim[1], 100))
# sys.exit()

xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 100), np.linspace(y_lim[0], y_lim[1], 100))
print(xx.ravel())
sys.exit()
Z = model.predict(np.c_xx.ravel()[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5, cmap='Dark2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression')
plt.show()
