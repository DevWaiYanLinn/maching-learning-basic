import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("data.csv")

# Extract the data columns
dates = df["Date"]
open_prices = df["Open"]
high_prices = df["High"]
low_prices = df["Low"]
close_prices = df["Close"]

# Create an array of indices for plotting
indices = np.arange(len(dates))
print(indices)

# Create a figure and axis for the plot
plt.figure(figsize=(12, 6))
plt.title("Stock Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price")

# Plotting the opening prices as blue lines
plt.plot(indices, open_prices, label="Open", color="blue", linestyle="-")

# Plotting the high prices as green dashed lines
plt.plot(indices, high_prices, label="High", color="green", linestyle="--")

# Plotting the low prices as red dotted lines
plt.plot(indices, low_prices, label="Low", color="red", linestyle=":")

# Plotting the closing prices as black solid lines
plt.plot(indices, close_prices, label="Close", color="black", linestyle="-")

# Adding x-axis labels with date values
plt.xticks(indices[::len(indices) // 10], dates[::len(indices) // 10], rotation=45)

# Adding a legend
plt.legend()

# Display the plot
plt.tight_layout()