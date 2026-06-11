import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Model/ebola_outcome_model.pkl")

st.title("🦠 Ebola Patient Outcome Predictor")

st.write(
    "Predict the likelihood of patient survival using admission features."
)

age = st.number_input("Age", min_value=0, max_value=120, value=30)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

fever = st.selectbox("Fever", [0, 1])
vomit = st.selectbox("Vomiting/Nausea", [0, 1])
diarrhoea = st.selectbox("Diarrhea", [0, 1])
fatigue = st.selectbox("Fatigue", [0, 1])
abdominal = st.selectbox("Abdominal Pain", [0, 1])
muscle = st.selectbox("Muscle Pain", [0, 1])
joint = st.selectbox("Joint Pain", [0, 1])
headache = st.selectbox("Headache", [0, 1])
breathing = st.selectbox("Breathing Difficulty", [0, 1])
rash = st.selectbox("Rash", [0, 1])
bleeding = st.selectbox("Bleeding", [0, 1])
contact = st.selectbox("Contact History", [0, 1])
funeral = st.selectbox("Funeral Exposure", [0, 1])
travel = st.selectbox("Travel History", [0, 1])

sex_value = 1 if sex == "Male" else 2

if st.button("Predict"):

    patient = pd.DataFrame([[
        age,
        sex_value,
        fever,
        vomit,
        diarrhoea,
        fatigue,
        abdominal,
        muscle,
        joint,
        headache,
        breathing,
        rash,
        bleeding,
        contact,
        funeral,
        travel
    ]], columns=[
        "age",
        "sex",
        "fever_ci",
        "vomit_nausea_ci",
        "diarrhoea_ci",
        "fatigue_ci",
        "abdominal_ci",
        "muscle_ci",
        "joint_ci",
        "headache_ci",
        "breathing_ci",
        "rash_ci",
        "bleeding_ci",
        "contact_ci",
        "funeral_ci",
        "travel_ci"
    ])

    probability = model.predict_proba(patient)[0][1]

    st.subheader(
        f"Predicted Survival Probability: {probability:.1%}"
    )

    if probability > 0.7:
        st.success("Low Mortality Risk")
    elif probability > 0.4:
        st.warning("Moderate Risk")
    else:
        st.error("High Mortality Risk")