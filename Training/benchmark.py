import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from model import create_model

# -------------------------------------------------------------
# 1. LOAD TEST DATASET
# -------------------------------------------------------------
print("=" * 65)
print("  FEDERATED LEARNING BENCHMARK & COMPARATIVE EVALUATION")
print("=" * 65)

X_test = pd.read_csv("X_test.csv").values
y_test = pd.read_csv("y_test.csv").values.ravel()
input_shape = X_test.shape[1]

num_clients = 4
client_data = []

for cid in range(1, num_clients + 1):
    Xc = pd.read_csv(f"client_{cid}_X.csv").values
    yc = pd.read_csv(f"client_{cid}_y.csv").values.ravel()
    client_data.append((Xc, yc))

def evaluate_model(model, X, y, threshold=0.7):
    preds_prob = model.predict(X, verbose=0).flatten()
    preds = (preds_prob >= threshold).astype(int)
    
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, preds_prob)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    }

results = []

# -------------------------------------------------------------
# 2. EVALUATE FEDERATED GLOBAL MODEL
# -------------------------------------------------------------
print("\n[1/3] Evaluating Federated Learning Model (FedAvg)...")
if os.path.exists("global_model.keras"):
    fed_model = tf.keras.models.load_model("global_model.keras")
    fed_metrics = evaluate_model(fed_model, X_test, y_test, threshold=0.7)
    fed_metrics["Model"] = "Federated FedAvg (Collaborative, Private)"
    results.append(fed_metrics)
    print(f"  Federated  -> Acc: {fed_metrics['Accuracy']:.4f} | Recall: {fed_metrics['Recall']:.4f} | AUC: {fed_metrics['ROC-AUC']:.4f}")
else:
    print("  Warning: global_model.keras not found, skipping FL evaluation.")

# -------------------------------------------------------------
# 3. TRAIN & EVALUATE CENTRALIZED MODEL (ALL DATA COMBINED)
# -------------------------------------------------------------
print("\n[2/3] Training Centralized Baseline Model (Pooled Data)...")
X_central = np.vstack([c[0] for c in client_data])
y_central = np.concatenate([c[1] for c in client_data])

classes = np.unique(y_central)
weights = compute_class_weight("balanced", classes=classes, y=y_central)
cw_dict = dict(zip(classes, weights))

central_model = create_model(input_shape)
central_model.fit(
    X_central,
    y_central,
    epochs=10,
    batch_size=128,
    class_weight=cw_dict,
    verbose=0
)
central_metrics = evaluate_model(central_model, X_test, y_test, threshold=0.7)
central_metrics["Model"] = "Centralized (Upper-Bound Baseline, No Privacy)"
results.append(central_metrics)
print(f"  Centralized -> Acc: {central_metrics['Accuracy']:.4f} | Recall: {central_metrics['Recall']:.4f} | AUC: {central_metrics['ROC-AUC']:.4f}")

# -------------------------------------------------------------
# 4. TRAIN & EVALUATE ISOLATED LOCAL MODELS
# -------------------------------------------------------------
print("\n[3/3] Training Isolated Local Models (Independent Clients)...")
local_metrics_list = []

for cid, (Xc, yc) in enumerate(client_data, start=1):
    c_classes = np.unique(yc)
    c_weights = compute_class_weight("balanced", classes=c_classes, y=yc)
    c_cw = dict(zip(c_classes, c_weights))
    
    loc_model = create_model(input_shape)
    loc_model.fit(
        Xc,
        yc,
        epochs=10,
        batch_size=128,
        class_weight=c_cw,
        verbose=0
    )
    m = evaluate_model(loc_model, X_test, y_test, threshold=0.7)
    local_metrics_list.append(m)
    print(f"  Client {cid} Isolated -> Acc: {m['Accuracy']:.4f} | Recall: {m['Recall']:.4f} | AUC: {m['ROC-AUC']:.4f}")

avg_local = {
    "Model": "Isolated Local Models (Average, No Collaboration)",
    "Accuracy": float(np.mean([m["Accuracy"] for m in local_metrics_list])),
    "Precision": float(np.mean([m["Precision"] for m in local_metrics_list])),
    "Recall": float(np.mean([m["Recall"] for m in local_metrics_list])),
    "F1-Score": float(np.mean([m["F1-Score"] for m in local_metrics_list])),
    "ROC-AUC": float(np.mean([m["ROC-AUC"] for m in local_metrics_list]))
}
results.append(avg_local)

# -------------------------------------------------------------
# 5. SUMMARY TABLE & EXPORT
# -------------------------------------------------------------
df_res = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
df_res.to_csv("benchmark_results.csv", index=False)

print("\n" + "=" * 80)
print("                           FINAL BENCHMARK RESULTS")
print("=" * 80)
print(df_res.to_string(index=False))
print("=" * 80)
print("Saved summary table to 'benchmark_results.csv'")

# -------------------------------------------------------------
# 6. GENERATE COMPARISON CHART
# -------------------------------------------------------------
metrics_to_plot = ["Accuracy", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics_to_plot))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#1f77b4", "#2ca02c", "#d62728"]

for i, row in df_res.iterrows():
    vals = [row[m] for m in metrics_to_plot]
    short_name = row["Model"].split("(")[0].strip()
    ax.bar(x + (i - 1) * width, vals, width, label=short_name, color=colors[i % len(colors)])

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Diabetes Risk Prediction: Centralized vs Isolated vs Federated", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=12)
ax.set_ylim([0.6, 1.02])
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.legend(loc="lower right", fontsize=10)

plt.tight_layout()
chart_path = "benchmark_comparison.png"
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Saved comparison figure to '{chart_path}'")
print("\nBenchmark completed successfully!")
