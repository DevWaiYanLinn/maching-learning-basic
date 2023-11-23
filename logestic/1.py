# from sklearn.linear_model import LogisticRegression
# import numpy as np
#
# X = np.array([[1, 2],
#               [2, 3],
#               [3, 4],
#               [4, 5],
#               [5, 6]])
# y = [0, 0, 1, 1, 1]
# model = LogisticRegression()
# model.fit(X, y)
#
# prediction = model.predict([[6, 7]])
# print(prediction)
#
# prediction2 = model.predict([[6, 7], [2, 1]])
# print(prediction2)

from sklearn.datasets import make_blobs, make_moons, make_circles
from matplotlib import pyplot as plt
from matplotlib import style

#
# style.use('fivethirtyeight')
#
# X, y = make_blobs(n_samples=100, centers=3, cluster_std=1, n_features=2)
#
# plt.scatter(X[:, 0], X[:, 1], s=20, color='g')
# plt.xlabel('X')
# plt.ylabel('Y')
#
# plt.show()
# plt.clf()

X, y = make_moons(n_samples=1000, noise=0.03)
plt.scatter(X[:, 0], X[:, 1], s=40, color='g')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
plt.clf()
