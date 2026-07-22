import pandas as pd
import tensorflow as tf

from joblib import load

# Load scaler
scaler = load("scaler.pkl")

# Load trained global model
model = tf.keras.models.load_model(
    "global_model.keras"
)

# Example Patient Data
patient = pd.DataFrame([{

    "gender": 1,                # Female=1, Male=0

    "age": 55,

    "hypertension": 1,

    "heart_disease": 0,

    "smoking_history": 3,       # former

    "bmi": 32.5,

    "HbA1c_level": 7.2,

    "blood_glucose_level": 180

}])

# Scale patient data
patient_scaled = scaler.transform(
    patient
)

# Predict
prediction = model.predict(
    patient_scaled,
    verbose=0
)

risk_score = float(
    prediction[0][0]
)

risk_percentage = (
    risk_score * 100
)

# Risk Level
if risk_percentage < 30:

    risk_level = "Low"

elif risk_percentage < 70:

    risk_level = "Moderate"

else:

    risk_level = "High"

print("\n===== DIABETES RISK ASSESSMENT =====\n")

print(
    f"Risk Score      : {risk_score:.4f}"
)

print(
    f"Risk Percentage : {risk_percentage:.2f}%"
)

print(
    f"Risk Level      : {risk_level}"
)

if risk_score >= 0.5:

    print(
        "\nPrediction      : Diabetes Risk Detected"
    )

else:

    print(
        "\nPrediction      : No Diabetes Risk"
    )