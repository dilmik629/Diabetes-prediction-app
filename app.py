import streamlit as st
import pandas as pd
import joblib

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Patient ගේ health details ටික දාන්න, diabetes risk එක predict කරගමු.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.divider()

if st.button("Predict", type="primary"):

    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High Risk: Diabetic ලෙස predict වෙනවා")
        st.write(f"Confidence: {probability[1]*100:.1f}%")
    else:
        st.success(f"✅ Low Risk: Non-Diabetic ලෙස predict වෙනවා")
        st.write(f"Confidence: {probability[0]*100:.1f}%")

    st.write("### Probability Breakdown")
    st.progress(float(probability[1]))
    st.caption(f"Diabetic Probability: {probability[1]*100:.1f}% | Non-Diabetic Probability: {probability[0]*100:.1f}%")