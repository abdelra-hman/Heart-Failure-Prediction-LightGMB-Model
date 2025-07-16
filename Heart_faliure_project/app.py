import streamlit as st
import numpy as np
import joblib

model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Failure Prediction", layout="centered")

st.title("❤️ Heart Failure Prediction")
st.markdown("أدخل بيانات المريض لتتنبأ بحدوث الوفاة.")

age = st.number_input("Age", 1, 120, 60)
anaemia = st.selectbox("Anaemia", ["No", "Yes"])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", 1, 10000, 200)
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
ejection_fraction = st.slider("Ejection Fraction", 10, 80, 40)
high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])
platelets = st.number_input("Platelets", 10000.0, 1000000.0, 250000.0)
serum_creatinine = st.number_input("Serum Creatinine", 0.1, 10.0, 1.2)
serum_sodium = st.number_input("Serum Sodium", 100, 150, 135)
sex = st.selectbox("Sex", ["Female", "Male"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
time = st.slider("Follow-up Time (days)", 0, 300, 100)

anaemia = 1 if anaemia == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
sex = 1 if sex == "Male" else 0
smoking = 1 if smoking == "Yes" else 0

platelets_per_age = platelets / (age + 1)
creatinine_per_ck = serum_creatinine / (creatinine_phosphokinase + 1)
ejection_per_age = ejection_fraction / (age + 1)
anaemia_creatinine = anaemia * serum_creatinine

features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                      high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                      sex, smoking, time,
                      platelets_per_age, creatinine_per_ck, ejection_per_age, anaemia_creatinine]])


if st.button("🔍 Predict"):
    scaled_input = scaler.transform(features)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 0:
        st.success(f"✅ Survived — Probability: {(1 - probability)*100:.2f}%")
    else:
        st.error(f"❌ Death Expected — Probability: {probability*100:.2f}%")
