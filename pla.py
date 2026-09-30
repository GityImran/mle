import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("pla_data.csv")

x1 = data["x1"].tolist()
x2 = data["x2"].tolist()
y = data["y"].tolist()

w1 = 0
w2 = 0
b = 0
eta = 1
epoch = 0

while True:
    epoch += 1
    errors = 0
    for i in range(len(y)):
        net = w1 * x1[i] + w2 * x2[i] + b
        if net >= 0:
            pred = 1
        else:
            pred = -1

        if pred != y[i]:
            w1 += eta * y[i] * x1[i]
            w2 += eta * y[i] * x2[i]
            b += eta * y[i]
            errors += 1

    print("Epoch", epoch, "Errors =", errors)
    if errors == 0:
        break

print("Final weights:", w1, w2)
print("Final bias:", b)

# --- Added Graph ---
colors = ['red' if val == 1 else 'blue' for val in y]
plt.scatter(x1, x2, c=colors)
if w2 != 0:
    x_vals = [min(x1) - 1, max(x1) + 1]
    y_vals = [-(w1/w2) * xi - (b/w2) for xi in x_vals]
    plt.plot(x_vals, y_vals, color='black', label="Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()