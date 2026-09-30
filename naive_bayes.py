import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("nb_data.csv")

features = ["Word_Free", "Word_Win", "Length", "Links"]
classes = data["Class"].unique()

test = [float(input("Enter Word_Free: ")),
        float(input("Enter Word_Win: ")),
        float(input("Enter Length: ")),
        float(input("Enter Links: "))]

def gaussian(x, mean, std):
    if std == 0:
        return 1 if x == mean else 0
    return (1 / (math.sqrt(2 * math.pi) * std)) * \
           math.exp(-((x - mean) ** 2) / (2 * std ** 2))

prob = {}

for c in classes:
    d = data[data["Class"] == c]
    p = len(d) / len(data)

    for i in range(4):
        mean = d[features[i]].mean()
        std = d[features[i]].std()
        p *= gaussian(test[i], mean, std)

    prob[c] = p

print("Probabilities:", prob)
print("Prediction:", max(prob, key=prob.get))

# Display graph
plt.bar(prob.keys(), prob.values())
plt.xlabel("Class")
plt.ylabel("Probability")
plt.title("Naive Bayes Classification")
plt.show()
