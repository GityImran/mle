import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("data1.csv")

x = data["ClassTest"].tolist()
y = data["Semester"].tolist()

print("Dataset")
print(data.to_string(index=False))

n = len(x)
xm = sum(x) / n
ym = sum(y) / n

def linear_regression(x, y, n, xm, ym):
    num = 0
    den = 0

    for i in range(n):
        num += (x[i] - xm) * (y[i] - ym)
        den += (x[i] - xm) ** 2

    w = num / den
    b = ym - w * xm

    yp = []

    for i in range(n):
        yp.append(w * x[i] + b)

    return w, b, yp

def calculate_errors(y, yp, n, ym):
    mae = 0
    mse = 0
    sse = 0
    sst = 0

    for i in range(n):
        error = y[i] - yp[i]
        mae += abs(error)
        mse += error ** 2
        sse += error ** 2
        sst += (y[i] - ym) ** 2

    mae = mae / n
    mse = mse / n
    rmse = math.sqrt(mse)
    r2 = 1 - sse / sst

    return mae, mse, rmse, r2

def display_graph(x, y, yp):
    plt.scatter(x, y, color="blue", label="Actual Data")
    plt.plot(x, yp, color="red", label="Regression Line")
    plt.xlabel("Class Test Marks")
    plt.ylabel("Semester Marks")
    plt.legend()
    plt.show()

w, b, yp = linear_regression(x, y, n, xm, ym)
mae, mse, rmse, r2 = calculate_errors(y, yp, n, ym)
prediction = w * 20 + b

print("Linear Regression")
print("-----------------")
print("Mean Class Test Marks =", xm)
print("Mean Semester Marks =", ym)
print("w =", w)
print("b =", round(b, 4))
print("MAE =", round(mae, 4))
print("MSE =", round(mse, 4))
print("RMSE =", round(rmse, 4))
print("R2 =", round(r2, 4))
print("Prediction for 20 =", round(prediction, 2))

display_graph(x, y, yp)