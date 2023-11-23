from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)

predictions = nb_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))
# plt.scatter(y_test, predictions, c=predictions)
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions)
plt.show()
