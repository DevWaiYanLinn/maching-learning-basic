import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_breast_cancer()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scalr = StandardScaler()
X_train_std = scalr.fit_transform(X_train)
X_test_std = scalr.transform(X_test)

def train_evaluate_model(X_train, y_train, X_test, y_test, C=1.0):
    model = LogisticRegression(C=C, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

model_good, accuracy_good = train_evaluate_model(X_train_std, y_train, X_test_std, y_test)
print(f'Accuracy with a good model: {accuracy_good}')

model_overfit, accuracy_overfit = train_evaluate_model(X_train_std, y_train, X_test_std, y_test, C=0.55)
print(f"Accuracy with an overfit model: {accuracy_overfit}")

model_underfit, accuracy_underfit = train_evaluate_model(X_train_std, y_train, X_test_std, y_test, C=100.0)
print(f"Accuracy with an underfit model: {accuracy_underfit}")
