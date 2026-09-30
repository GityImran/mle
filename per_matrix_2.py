import random
import matplotlib.pyplot as plt

# 1. Generate 1000 actual values
y_true = [0] * 900 + [1] * 100

# Generate 1000 predicted values
y_pred = [0] * 900 + [1] * 100

# Shuffle both randomly
random.shuffle(y_true)
random.shuffle(y_pred)

# 2. Calculate confusion matrix
TP = 0
TN = 0
FP = 0
FN = 0

for i in range(1000):
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

# 3. Calculate precision and recall
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

if FP > FN:
    print("Precision is more important because FP is higher.")
else:
    print("Recall is more important because FN is higher.")

# 4. Randomly give priority to FP or FN
priority = random.choice(["FP", "FN"])
if priority == "FP":
    beta = 0.5
else:
    beta = 2

print("Priority given to:", priority)
print("Beta =", beta)

# 5. Calculate F-beta score
try:
    f_beta = ((1 + beta ** 2) * precision * recall) / ((beta ** 2 * precision) + recall)
except ZeroDivisionError:
    f_beta = 0
print("F-beta score =", f_beta)

# --- Added Graph ---
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap='Blues')
plt.title('Confusion Matrix\n')
fig.colorbar(cax)
ax.set_xticklabels([''] + ['Predicted 0', 'Predicted 1'])
ax.set_yticklabels([''] + ['Actual 0', 'Actual 1'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='red')
plt.show()