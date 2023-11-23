import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

# Load the dataset from 'data.csv'
parser = lambda date: pd.datetime.strptime(date, '%d%b%Y')
df = pd.read_csv('data.csv',parse_dates=['Date'])
df = df[['Date','Open']]
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, cmap=cm.RdYlGn)
plt.show()

sys.exit

# Data Cleaning: Check for missing values and remove them if necessary
df.dropna(inplace=True)

# Feature Selection: Choose the features and target variable
X = df[['Low', 'High', 'Open']].values
y = df[['Close']].values

# Feature Scaling: Scale the features
X = scale(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_predict = model.predict(X_test)

# Calculate Mean Squared Error
msc = mean_squared_error(y_test, y_predict)
print(f'Mean square error {msc}')

# Visualize the predictions
plt.scatter(y_test, y_predict)
print(f'Actual Close: {y_test[0]}, Predicted Close: {y_predict[0]}')

# Display the plot
plt.show()

# Optionally, you can save the model for future use
# import joblib
# joblib.dump(model, 'linear_regression_model.pkl')

# Exit the program
sys.exit()
