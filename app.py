import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("heart_disease_rf_model.pkl")

# Title
st.title("🏥 Heart Disease Risk Prediction")
st.write("Enter patient information to assess heart disease risk.")

# Mappings (convert categorical inputs to numbers if your model was trained with encoded values)
sex_map = {"male": 1, "female": 0}
cp_map = {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}
fbs_map = {"True": 1, "False": 0}
restecg_map = {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
thal_map = {"normal": 1, "fixed defect": 2, "reversible defect": 3}

# Input form
with st.form("patient_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", list(sex_map.keys()))
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_map.keys()))
    restecg = st.selectbox("Resting ECG", list(restecg_map.keys()))
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", list(exang_map.keys()))
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", list(slope_map.keys()))
    ca = st.selectbox("Number of major vessels (0–3) colored by fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", list(thal_map.keys()))

    submit = st.form_submit_button("Predict")

if submit:
    # Prepare input data with correct column names and encodings
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex_map[sex],
        "cp": cp_map[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_map[fbs],
        "restecg": restecg_map[restecg],
        "thalch": thalach,  # must match the trained model's column name
        "exang": exang_map[exang],
        "oldpeak": oldpeak,
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal_map[thal]
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    # Output result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Confidence: {1 - proba:.2f})")
