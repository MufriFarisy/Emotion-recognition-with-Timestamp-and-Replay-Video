import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# baca hasil pengujian
df = pd.read_csv("hasil_pengujian.csv")

y_true = df['actual']
y_pred = df['predicted']

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# accuracy
acc = accuracy_score(y_true, y_pred)
print("\nAccuracy:", acc)

# precision, recall, f1-score
print("\nClassification Report:")
print(classification_report(y_true, y_pred))