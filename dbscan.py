import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("dbscan_data.csv")

points = data["Point"].tolist()
x = data["x"].tolist()
y = data["y"].tolist()

eps = 2
minpts = 2

clusters = []
noise = []

for i in range(len(x)):
    neighbours = []
    for j in range(len(x)):
        distance = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
        if distance <= eps and i != j:
            neighbours.append(j)

    if len(neighbours) >= minpts:
        cluster = [points[i]]
        for j in neighbours:
            cluster.append(points[j])
        clusters.append(cluster)
    else:
        noise.append(points[i])

print("Clusters:", clusters)
print("Noise:", noise)

# --- Added Graph ---
plotted_points = set()
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

# Plot clusters
for c_idx, cluster in enumerate(clusters):
    c_color = colors[c_idx % len(colors)]
    for pt in cluster:
        if pt not in plotted_points:
            idx = points.index(pt)
            plt.scatter(x[idx], y[idx], color=c_color)
            plotted_points.add(pt)

# Plot noise
for pt in noise:
    if pt not in plotted_points:
        idx = points.index(pt)
        plt.scatter(x[idx], y[idx], color='black', marker='x', label='Noise' if 'Noise' not in plt.gca().get_legend_handles_labels()[1] else "")
        plotted_points.add(pt)

plt.title("DBSCAN Results")
plt.show()