import streamlit as st
import pandas as pd
import joblib

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("Model/ebola_outcome_model.pkl")

# Hardcoded feature order (REMOVES ALL FILE ERRORS)
features = [
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
]

# -------------------------
# APP TITLE
# -------------------------
st.title("🦠 Ebola Patient Outcome Predictor")
st.write("Predicts survival likelihood based on clinical admission features.")

# -------------------------
# INPUTS
# -------------------------
age = st.number_input("Age", 0, 120, 30)

sex = st.selectbox("Sex", ["Male", "Female"])
sex_value = 1 if sex == "Male" else 0

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

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict"):

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex_value,
        "fever_ci": fever,
        "vomit_nausea_ci": vomit,
        "diarrhoea_ci": diarrhoea,
        "fatigue_ci": fatigue,
        "abdominal_ci": abdominal,
        "muscle_ci": muscle,
        "joint_ci": joint,
        "headache_ci": headache,
        "breathing_ci": breathing,
        "rash_ci": rash,
        "bleeding_ci": bleeding,
        "contact_ci": contact,
        "funeral_ci": funeral,
        "travel_ci": travel
    }])

    # Ensure correct feature order
    patient = patient.reindex(columns=features)

    # Model prediction
    proba = model.predict_proba(patient)[0]
    classes = model.classes_

    survival_index = list(classes).index(1) if 1 in classes else 0
    survival_prob = proba[survival_index]

    # -------------------------
    # SYMPTOM SCORE
    # -------------------------
    symptom_score = sum([
        fever, vomit, diarrhoea, fatigue, abdominal,
        muscle, joint, headache, breathing, rash, bleeding
    ])

    all_zero = symptom_score == 0

    # -------------------------
    # OUTPUT
    # -------------------------
    st.subheader("Prediction Result")

    # CASE 1: ALL SYMPTOMS = 0
    if all_zero:
        st.info("No symptoms reported. Prediction reflects baseline dataset behavior.")
        st.write("No survival probability displayed (baseline case).")

        adjusted_score = (survival_prob - 0.5) * 2
        adjusted_score = max(0, min(1, adjusted_score))

        st.write(f"Baseline Index (interpreted): {adjusted_score:.2f}")

    # CASE 2: ANY SYMPTOMS PRESENT
    else:
        st.write(f"Survival Probability: {survival_prob:.1%}")

        if survival_prob >= 0.65:
            st.success("High Survival Likelihood")
            st.write("Model indicates low risk based on learned patterns.")

        elif survival_prob >= 0.45:
            st.warning("Moderate / Uncertain Outcome")
            st.write("Model shows mixed indicators; uncertainty is present.")

        else:
            st.error("High Risk Outcome")
            st.write("Model indicates elevated risk based on symptom pattern.")

        st.progress(float(survival_prob))

    # -------------------------
    # DISCLAIMER
    # -------------------------
    st.caption(
        "Note: This tool is a statistical learning model and not a medical diagnostic system."
    )

    # -------------------------
    # DEBUG (PROFESSOR VIEW)
    # -------------------------
    with st.expander("Model Details (for review)"):
        st.write("Model classes:", model.classes_)
        st.write("Feature order:", features)
        st.write("Raw probabilities:", proba)
        st.write("Symptom score:", symptom_score)
