import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("performance_data.csv")

# Get actual and predicted values
y_true = df["Actual"].tolist()
y_pred = df["Predicted"].tolist()

# Counters for the four cases
TP = 0 # Actual 1, Predicted 1
TN = 0 # Actual 0, Predicted 0
FP = 0 # Actual 0, Predicted 1
FN = 0 # Actual 1, Predicted 0

# Compare actual and predicted values
for i in range(len(y_true)):
    if y_true[i] == 1 and y_pred[i] == 1:
        TP += 1
    elif y_true[i] == 0 and y_pred[i] == 0:
        TN += 1
    elif y_true[i] == 0 and y_pred[i] == 1:
        FP += 1
    elif y_true[i] == 1 and y_pred[i] == 0:
        FN += 1

print("Confusion Matrix:")
cm = [[TN, FP],
      [FN, TP]]
print(cm)

accuracy = (TP + TN) / len(y_true)
print("Accuracy =", accuracy)
print("Accuracy =", accuracy * 100, "%")

# --- Added Graph ---
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap='Blues')
plt.title('Confusion Matrix\n')
fig.colorbar(cax)
ax.set_xticklabels([''] + ['Predicted 0', 'Predicted 1'])
ax.set_yticklabels([''] + ['Actual 0', 'Actual 1'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center')
plt.show()