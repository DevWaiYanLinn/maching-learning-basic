# Plotting with Matplotlib
import pandas as pd
import  sys

djia_data = pd.read_csv('data.csv')
# djia_data.head()

# data cleaning
djia_data = djia_data.rename(columns={' Open': 'Open', ' High': 'High', ' Low': 'Low', ' Close': 'Close'})

# ensure that the Date variable is a datetime variable and sort
djia_data['Date'] = pd.to_datetime(djia_data['Date'])
djia_data = djia_data.sort_values(by='Date')
# djia_data.sort_values(by='Date', inplace=True)
# index=False argument - prevent writing the DataFrame index to the CSV file
# djia_data.to_csv('/content/drive/MyDrive/AI_Course_Docs/Metro/HistoricalPrices(2022-23).csv', index=False)

### Basics of Matplotlib ######
# loading Matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# plt.plot(djia_data['Date'], djia_data['Close'])
# plt.show()
# # sys.exit()
# plt.plot(djia_data['Date'], djia_data['Open'])
# plt.plot(djia_data['Date'], djia_data['Close'])
# plt.show()# adding a legend
# plt.plot(djia_data['Date'], djia_data['Open'], label='Open')
# plt.plot(djia_data['Date'], djia_data['Close'], label='Close')
# plt.legend()
# plt.show()
#
# # draw barplot
# # Import the calendar package
from calendar import month_name
#
# # Order by months by chronological order
djia_data['Month'] = pd.Categorical(djia_data['Date'].dt.month_name(), month_name[1:])

# Group metrics by monthly averages
djia_monthly_mean = djia_data \
    .groupby('Month') \
    .mean() \
    .reset_index()

djia_monthly_mean.head(6)

# vertical barplot
plt.bar(djia_monthly_mean['Month'], height=djia_monthly_mean['Close'])
plt.show()
#
# # reorderbars in bar plot
# djia_monthly_mean_srtd = djia_monthly_mean.sort_values(by='Close', ascending=False)
#
# plt.bar(djia_monthly_mean_srtd['Month'], height=djia_monthly_mean_srtd['Close'], width=-0.4)
# plt.show()
#
# # horizontal bar plot
# plt.barh(djia_monthly_mean_srtd['Month'], height=djia_monthly_mean_srtd['Close'], width=-0.4)
# plt.show()
#
# # scatter plot
# plt.scatter(djia_data['Open'], djia_data['Close'])
# plt.show()

# scatter plot with a trend line
z = np.polyfit(djia_data['Open'], djia_data['Close'], 1)
p = np.poly1d(z)

plt.scatter(djia_data['Open'], djia_data['Close'])
plt.plot(djia_data['Open'], p(djia_data['Open']))
plt.show()

# setting the Plot Title and Axis Labels
plt.scatter(djia_data['Open'], djia_data['Close'])
plt.show()

# change colors
plt.plot(djia_data['Date'], djia_data['Open'], color='black')
plt.plot(djia_data['Date'], djia_data['Close'], color='red')
plt.show()

# bars
plt.bar(djia_monthly_mean_srtd['Month'], height=djia_monthly_mean_srtd['Close'],
        color=['blue', 'gray', 'gray', 'gray', 'gray', 'gray'])
plt.show()

# points
plt.scatter(djia_data[djia_data['Month'] == 'January']['Open'], djia_data[djia_data['Month'] == 'January']['Close'],
            color='blue')

plt.scatter(djia_data[djia_data['Month'] != 'January']['Open'], djia_data[djia_data['Month'] != 'January']['Close'],
            color='gray')

plt.show()

# using colormaps
plt.scatter(djia_data['Open'], djia_data['Close'], c=djia_data['Close'], cmap=plt.cm.plasma)

plt.show()

# Setting Axis Limits - plt.xlim() and plt.ylim()

# saving plots
plt.scatter(djia_data['Open'], djia_data['Close'])
plt.savefig('DJIA 2022 Scatterplot Open vs. Close.png')

