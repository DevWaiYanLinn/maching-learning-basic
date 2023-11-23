import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
# plt.show()

data = list(zip(x, y))
inertias = []

print(type(list(zip(x, y))[0]))

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

print(inertias)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Inertia for different values of K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.close()

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# plt.scatter(x, y, c=kmeans.labels_)
