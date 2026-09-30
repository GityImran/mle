import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("and_gate.csv")

X = data[["x1", "x2"]].values.tolist()
y = data["y"].tolist()

def train(X, y):
    w1 = 0
    w2 = 0
    b = -1
    eta = 1

    while True:
        errors = 0
        for i in range(len(X)):
            net = w1 * X[i][0] + w2 * X[i][1] + b
            if net >= 0:
                pred = 1
            else:
                pred = 0

            if pred != y[i]:
                w1 += eta * (y[i] - pred) * X[i][0]
                w2 += eta * (y[i] - pred) * X[i][1]
                b += eta * (y[i] - pred)
                errors += 1

        if errors == 0:
            break
    return w1, w2, b

def test(X, w1, w2, b):
    for x in X:
        net = w1 * x[0] + w2 * x[1] + b
        if net >= 0:
            pred = 1
        else:
            pred = 0
        print(x, "->", pred)

w1, w2, b = train(X, y)

print("Final weights:", w1, w2)
print("Final bias:", b)
print("AND Gate Output:")
test(X, w1, w2, b)

# --- Added Graph ---
x1_vals = [pt[0] for pt in X]
x2_vals = [pt[1] for pt in X]
colors = ['green' if val == 1 else 'red' for val in y]

plt.scatter(x1_vals, x2_vals, c=colors, s=100)
if w2 != 0:
    xv = [min(x1_vals) - 0.5, max(x1_vals) + 0.5]
    yv = [-(w1/w2) * xi - (b/w2) for xi in xv]
    plt.plot(xv, yv, color='black', label="Decision Boundary")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()