import gradio as gr
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define prediction function
def predict_heart_failure(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                          sex, smoking, time):

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
    return "✅ Survived" if prediction == 0 else "❌ Death Expected"

# Gradio interface
inputs = [
    gr.Number(label="Age"),
    gr.Radio([0, 1], label="Anaemia (0=No, 1=Yes)"),
    gr.Number(label="Creatinine Phosphokinase"),
    gr.Radio([0, 1], label="Diabetes (0=No, 1=Yes)"),
    gr.Number(label="Ejection Fraction"),
    gr.Radio([0, 1], label="High Blood Pressure"),
    gr.Number(label="Platelets"),
    gr.Number(label="Serum Creatinine"),
    gr.Number(label="Serum Sodium"),
    gr.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
    gr.Radio([0, 1], label="Smoking (0=No, 1=Yes)"),
    gr.Number(label="Follow-up Time"),
]

app = gr.Interface(fn=predict_heart_failure, inputs=inputs, outputs="text",
                   title="Heart Failure Prediction",
                   description="Predicts if a heart failure patient is likely to die (DEATH_EVENT) based on medical data.")

app.launch()
