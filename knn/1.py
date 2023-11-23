
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(1)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_predict = neigh.predict(X_test)
asc = accuracy_score(y_test, y_predict)
print(asc)