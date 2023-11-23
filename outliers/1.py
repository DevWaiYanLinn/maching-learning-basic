import sys

import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
sns.boxplot(x="total_bill", data=tips)
plt.xlabel('Day of the week')
plt.ylabel('Total Bill Amount')
plt.title('Boxplot of Total Bill Amount by Day')

plt.show()

Q1 = tips['total_bill'].quantile(0.25)
Q3 = tips['total_bill'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = tips[(tips['total_bill'] < lower_bound) | (tips['total_bill'] > upper_bound)]
print(outliers)
