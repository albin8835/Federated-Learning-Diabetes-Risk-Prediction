import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import load

FEATURE_NAMES = [
    "Gender",
    "Age",
    "Hypertension",
    "Heart Disease",
    "Smoking History",
    "BMI",
    "HbA1c Level",
    "Blood Glucose"
]

def load_explainer_assets():
    model_path = "global_model.keras" if os.path.exists("global_model.keras") else "../Training/global_model.keras"
    scaler_path = "scaler.pkl" if os.path.exists("scaler.pkl") else "../Training/scaler.pkl"
    
    model = tf.keras.models.load_model(model_path)
    scaler = load(scaler_path)
    return model, scaler

def compute_feature_contributions(model, scaler, patient_raw_features):
    """
    Computes feature importance for a single patient using sensitivity/perturbation analysis
    relative to a standardized population baseline. Fast, deterministic, and lightweight.
    """
    patient_raw = np.array(patient_raw_features, dtype=float).reshape(1, -1)
    patient_scaled = scaler.transform(patient_raw)
    base_prob = float(model.predict(patient_scaled, verbose=0)[0][0])
    
    # Population median reference (zeros in standardized space)
    zero_baseline = np.zeros_like(patient_scaled)
    
    contributions = {}
    for i, name in enumerate(FEATURE_NAMES):
        # Perturb single feature towards neutral baseline
        perturbed = patient_scaled.copy()
        perturbed[0, i] = zero_baseline[0, i]
        perturbed_prob = float(model.predict(perturbed, verbose=0)[0][0])
        
        # Marginal impact: how much did this feature shift the risk?
        impact = base_prob - perturbed_prob
        contributions[name] = impact
        
    total_abs_impact = sum(abs(v) for v in contributions.values())
    if total_abs_impact == 0:
        total_abs_impact = 1e-6
        
    percentage_impact = {
        name: (abs(val) / total_abs_impact) * 100
        for name, val in contributions.items()
    }
    
    # Sort by impact
    sorted_contributions = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )
    
    return base_prob, sorted_contributions, percentage_impact

def explain_and_plot(patient_features, save_path="patient_explanation.png"):
    model, scaler = load_explainer_assets()
    prob, sorted_impacts, percentages = compute_feature_contributions(model, scaler, patient_features)
    
    print("=" * 65)
    print("      PATIENT EXPLAINABLE AI (XAI) RISK ANALYSIS")
    print("=" * 65)
    print(f"Overall Predicted Diabetes Risk: {prob * 100:.2f}%\n")
    print(f"{'Feature':<20} | {'Impact on Risk':<15} | {'Relative Importance':<20}")
    print("-" * 65)
    
    for feat, impact in sorted_impacts:
        direction = "Increases Risk (+)" if impact >= 0 else "Lowers Risk (-)"
        pct = percentages[feat]
        print(f"{feat:<20} | {direction:<15} | {pct:>6.2f}%")
        
    print("=" * 65)
    
    # Generate visual bar chart
    feats = [item[0] for item in reversed(sorted_impacts)]
    pcts = [percentages[item[0]] for item in reversed(sorted_impacts)]
    colors = ["#d62728" if dict(sorted_impacts)[f] >= 0 else "#2ca02c" for f in feats]
    
    plt.figure(figsize=(9, 5))
    bars = plt.barh(feats, pcts, color=colors, edgecolor="black", height=0.6)
    plt.xlabel("Contribution to Risk Prediction (%)", fontsize=11, fontweight="bold")
    plt.title(f"Explainable AI: Key Drivers of Predicted Risk ({prob*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlim(0, max(pcts) + 10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=10)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Explanation plot saved to '{save_path}'")
    return prob, sorted_impacts, percentages

if __name__ == "__main__":
    # High-risk example patient
    # [gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c, glucose]
    sample_patient = [1, 55, 1, 0, 3, 32.5, 7.2, 180]
    explain_and_plot(sample_patient)
