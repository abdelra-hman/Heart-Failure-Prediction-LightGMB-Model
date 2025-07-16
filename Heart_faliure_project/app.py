import gradio as gr
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prediction function
def predict_heart_failure(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                          sex, smoking, time):

    # Encode categorical values
    anaemia = 1 if anaemia == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
    smoking = 1 if smoking == "Yes" else 0
    sex = 1 if sex == "Male" else 0

    # Feature engineering
    platelets_per_age = platelets / (age + 1)
    creatinine_per_ck = serum_creatinine / (creatinine_phosphokinase + 1)
    ejection_per_age = ejection_fraction / (age + 1)
    anaemia_creatinine = anaemia * serum_creatinine

    # Create input array
    features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                          sex, smoking, time,
                          platelets_per_age, creatinine_per_ck, ejection_per_age, anaemia_creatinine]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][prediction]

    result = "✅ Survived" if prediction == 0 else "❌ Death Expected"
    return f"{result} — Probability: {proba:.2%}"

# Input components
inputs = [
    gr.Number(label="Age"),
    gr.Radio(["No", "Yes"], label="Anaemia"),
    gr.Number(label="Creatinine Phosphokinase"),
    gr.Radio(["No", "Yes"], label="Diabetes"),
    gr.Number(label="Ejection Fraction"),
    gr.Radio(["No", "Yes"], label="High Blood Pressure"),
    gr.Number(label="Platelets"),
    gr.Number(label="Serum Creatinine"),
    gr.Number(label="Serum Sodium"),
    gr.Radio(["Female", "Male"], label="Sex"),
    gr.Radio(["No", "Yes"], label="Smoking"),
    gr.Number(label="Follow-up Time"),
]

# Example patients
examples = [
    [60, "No", 250, "Yes", 38, "No", 250000, 1.4, 130, "Male", "Yes", 90],
    [45, "Yes", 150, "No", 50, "Yes", 180000, 1.0, 138, "Female", "No", 120],
    [65, "Yes", 400, "Yes", 30, "Yes", 300000, 2.5, 120, "Male", "Yes", 50],
    [50, "No", 100, "No", 45, "No", 220000, 1.2, 140, "Female", "No", 100],
]

# Gradio app
app = gr.Interface(
    fn=predict_heart_failure,
    inputs=inputs,
    outputs="text",
    title="🫀 Heart Failure Prediction",
    description="Predict heart failure risk using clinical data (LightGBM model).",
    examples=examples
)

app.launch()
