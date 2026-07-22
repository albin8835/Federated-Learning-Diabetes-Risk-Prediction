import pandas as pd
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model(
    "global_model.keras"
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

X_test = pd.read_csv(
    "X_test.csv"
).values

y_test = pd.read_csv(
    "y_test.csv"
).values.ravel()

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("Positive cases:", np.sum(y_test))
print("Negative cases:", len(y_test) - np.sum(y_test))

model = tf.keras.models.load_model(
    "global_model.keras"
)

pred_prob = model.predict(X_test, verbose=0).flatten()

pred = (pred_prob > 0.7).astype(int)

print("\nProbability Statistics")
print("Min :", pred_prob.min())
print("Max :", pred_prob.max())
print("Mean:", pred_prob.mean())

print("\nPredicted Positives:", np.sum(pred))
print("Actual Positives   :", np.sum(y_test))

print("\nProbability Quartiles")
print(
    np.percentile(
        pred_prob,
        [0,25,50,75,90,95,99,100]
    )
)

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred
)

recall = recall_score(
    y_test,
    pred
)

f1 = f1_score(
    y_test,
    pred
)

auc = roc_auc_score(
    y_test,
    pred_prob
)

cm = confusion_matrix(
    y_test,
    pred
)

print("\n===== RESULTS =====")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"AUC      : {auc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)