import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

x = data["Marks"].tolist()
y = data["y"].tolist()

y = [1 if i == 1 else 0 for i in y]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def hypothesis(x, w, b):
    return sigmoid(w * x + b)

def cost(x, y, w, b):
    total = 0
    for i in range(len(x)):
        p = hypothesis(x[i], w, b)
        total += -(y[i] * math.log(p) + (1-y[i]) * math.log(1-p))
    return total / len(x)

def gradient_descent(x, y, w, b):
    lr = 0.0001
    epochs = 10000
    n = len(x)
    for j in range(epochs):
        dw = 0
        db = 0
        for i in range(n):
            p = hypothesis(x[i], w, b)
            error = p - y[i]
            dw += error * x[i]
            db += error
        dw = dw / n
        db = db / n
        w = w - lr * dw
        b = b - lr * db
    return w, b

def logistic_regression(x, y):
    w = 0
    b = 0
    w, b = gradient_descent(x, y, w, b)
    return w, b

w, b = logistic_regression(x, y)
c = cost(x, y, w, b)

marks = 20
probability = hypothesis(marks, w, b)

if probability >= 0.5:
    result = "PASS"
else:
    result = "FAIL"

print("Logistic Regression")
print("w =", round(w, 4))
print("b =", round(b, 4))
print("Cost =", round(c, 4))
print("Marks =", marks)
print("Probability =", round(probability, 4))
print("Prediction =", result)

# --- Added Graph ---
plt.scatter(x, y, color="blue", label="Actual Data")
sorted_x = sorted(x)
probs = [hypothesis(i, w, b) for i in sorted_x]
plt.plot(sorted_x, probs, color="red", label="Sigmoid Curve")
plt.scatter([marks], [probability], color="green", marker="X", s=100, label="Test Prediction")
plt.xlabel("Marks")
plt.ylabel("Probability / Class")
plt.legend()
plt.show()