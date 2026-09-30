import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("kmeans_data.csv")

x = data["x"].tolist()
y = data["y"].tolist()
points = data["Point"].tolist()

k = 3

c1 = [x[0], y[0]]
c2 = [x[1], y[1]]
c3 = [x[2], y[2]]

for iteration in range(10):
    cluster1 = []
    cluster2 = []
    cluster3 = []

    for i in range(len(x)):
        d1 = math.sqrt((x[i] - c1[0]) ** 2 + (y[i] - c1[1]) ** 2)
        d2 = math.sqrt((x[i] - c2[0]) ** 2 + (y[i] - c2[1]) ** 2)
        d3 = math.sqrt((x[i] - c3[0]) ** 2 + (y[i] - c3[1]) ** 2)

        if d1 <= d2 and d1 <= d3:
            cluster1.append(i)
        elif d2 <= d1 and d2 <= d3:
            cluster2.append(i)
        else:
            cluster3.append(i)

    c1 = [sum(x[i] for i in cluster1) / len(cluster1),
          sum(y[i] for i in cluster1) / len(cluster1)]

    c2 = [sum(x[i] for i in cluster2) / len(cluster2),
          sum(y[i] for i in cluster2) / len(cluster2)]

    c3 = [sum(x[i] for i in cluster3) / len(cluster3),
          sum(y[i] for i in cluster3) / len(cluster3)]

print("Cluster 1:", [points[i] for i in cluster1])
print("Cluster 2:", [points[i] for i in cluster2])
print("Cluster 3:", [points[i] for i in cluster3])
print("Centroid 1:", c1)
print("Centroid 2:", c2)
print("Centroid 3:", c3)

# --- Added Graph ---
for idx in cluster1: plt.scatter(x[idx], y[idx], color='red')
for idx in cluster2: plt.scatter(x[idx], y[idx], color='green')
for idx in cluster3: plt.scatter(x[idx], y[idx], color='blue')

# Plot the centroids
plt.scatter([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color='black', marker='X', s=100, label='Centroids')
plt.title("K-Means Clustering Result")
plt.legend()
plt.show()