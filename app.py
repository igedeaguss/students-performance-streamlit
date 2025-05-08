import streamlit as st
import numpy as np
import joblib

# Load model dan scaler (jika tersedia)
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")  

# --- Dictionary Encoding ---
marital_status_dict = {
    'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4,
    'Facto Union': 5, 'Legally Seperated': 6
}

gender_dict = {'Female': 0, 'Male': 1}

application_mode_dict = {
    '1st Phase - General Contingent': 1,
    'Ordinance No. 612/93': 2,
    '1st Phase - Special Contingent (Azores Island)': 5,
    'Holders of Other Higher Courses': 7,
    'Ordinance No. 854-B/99': 10,
    'International Student (Bachelor)': 15,
    '1st Phase - Special Contingent (Madeira Island)': 16,
    '2nd Phase - General Contingent': 17,
    '3rd Phase - General Contingent': 18,
    'Ordinance No. 533-A/99, Item B2 (Different Plan)': 26,
    'Ordinance No. 533-A/99, Item B3 (Other Institution)': 27,
    'Over 23 Years Old': 39,
    'Transfer': 42,
    'Change of Course': 43,
    'Technological Specialization Diploma Holders': 44,
    'Change of Institution/Course': 51,
    'Short Cycle Diploma Holders': 53,
    'Change of Institution/Course (International)': 57
}

st.title("Klasifikasi Performa Mahasiswa")

# --- Form Input ---
marital_status = st.selectbox("Marital Status", list(marital_status_dict.keys()))
application_mode = st.selectbox("Application Mode", list(application_mode_dict.keys()))
prev_qual_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0)
admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0)
displaced = st.radio("Displaced", ["Yes", "No"])
debtor = st.radio("Debtor", ["Yes", "No"])
tuition_up_to_date = st.radio("Tuition Fees Up To Date", ["Yes", "No"])
gender = st.radio("Gender", ["Male", "Female"])
scholarship = st.radio("Scholarship Holder", ["Yes", "No"])
age = st.number_input("Age at Enrollment", min_value=15, max_value=100)

cu_1st_enrolled = st.number_input("Curricular Units 1st Sem Enrolled", min_value=0)
cu_1st_approved = st.number_input("Curricular Units 1st Sem Approved", min_value=0)
cu_1st_grade = st.number_input("Curricular Units 1st Sem Grade", min_value=0.0, max_value=20.0)

cu_2nd_enrolled = st.number_input("Curricular Units 2nd Sem Enrolled", min_value=0)
cu_2nd_evals = st.number_input("Curricular Units 2nd Sem Evaluations", min_value=0)
cu_2nd_approved = st.number_input("Curricular Units 2nd Sem Approved", min_value=0)
cu_2nd_grade = st.number_input("Curricular Units 2nd Sem Grade", min_value=0.0, max_value=20.0)
cu_2nd_wo_evals = st.number_input("Curricular Units 2nd Sem Without Evaluations", min_value=0)

# --- Encoding ---
def encode_input():
    return np.array([[
        marital_status_dict[marital_status],
        application_mode_dict[application_mode],
        prev_qual_grade,
        admission_grade,
        1 if displaced == "Yes" else 0,
        1 if debtor == "Yes" else 0,
        1 if tuition_up_to_date == "Yes" else 0,
        gender_dict[gender],
        1 if scholarship == "Yes" else 0,
        age,
        cu_1st_enrolled,
        cu_1st_approved,
        cu_1st_grade,
        cu_2nd_enrolled,
        cu_2nd_evals,
        cu_2nd_approved,
        cu_2nd_grade,
        cu_2nd_wo_evals
    ]])

# --- Predict Button ---
if st.button("Prediksi"):
    input_data = encode_input()
    input_scaled = scaler.transform(input_data)  # Uncomment jika ada scaler
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Mahasiswa diprediksi *Dropout*.")
    else:
        st.success("Mahasiswa diprediksi *Lulus*.")
