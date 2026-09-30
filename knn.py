import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("knn_data.csv")

x = data["X"].tolist()
y = data["Y"].tolist()
label = data["Label"].tolist()

print("Dataset")
print(data)

test_x = float(input("Enter X: "))
test_y = float(input("Enter Y: "))
k = int(input("Enter k: "))

distances = []

for i in range(len(x)):
    d = math.sqrt((test_x - x[i]) ** 2 + (test_y - y[i]) ** 2)
    distances.append((d, label[i]))

distances.sort()
nearest = distances[:k]

red = 0
blue = 0
green = 0

for d, lbl in nearest:
    if lbl == "Red":
        red += 1
    elif lbl == "Blue":
        blue += 1
    else:
        green += 1

if red > blue and red > green:
    prediction = "Red"
elif blue > red and blue > green:
    prediction = "Blue"
else:
    prediction = "Green"

print("Test Point:", (test_x, test_y))
print("K =", k)
print("Nearest Neighbours:", nearest)
print("Prediction:", prediction)

# --- Added Graph ---
# Plot all known points in their respective colors
colors = [l.lower() for l in label]
plt.scatter(x, y, c=colors)

# Plot the new test point with an X
plt.scatter(test_x, test_y, c=prediction.lower(), marker='X', s=200, edgecolors='black', label="Test Point")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"KNN Classification (k={k})")
plt.legend()
plt.show()