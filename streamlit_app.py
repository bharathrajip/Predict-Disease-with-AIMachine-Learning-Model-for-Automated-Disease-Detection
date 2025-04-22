
import streamlit as st
import numpy as np
import joblib

# Model & Scaler load பண்ணுறது
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Disease Prediction App")
st.write("நீங்கள் அனுபவிக்கும் அறிகுறிகளை உள்ளிடவும்:")

# Top 20 symptom-based inputs (mostly binary 0 or 1)
input_features = {
    "high_fever": st.checkbox("உயர் காய்ச்சல் (High Fever)"),
    "vomiting": st.checkbox("வாந்தி (Vomiting)"),
    "headache": st.checkbox("தலைவலி (Headache)"),
    "fatigue": st.checkbox("களைப்பு (Fatigue)"),
    "cough": st.checkbox("இருமல் (Cough)"),
    "muscle_pain": st.checkbox("தசை வலி (Muscle Pain)"),
    "nausea": st.checkbox("மனவருத்தம் (Nausea)"),
    "abdominal_pain": st.checkbox("வயிற்று வலி (Abdominal Pain)"),
    "chills": st.checkbox("அறைஅறுப்பு (Chills)"),
    "loss_of_appetite": st.checkbox("பசிக்குறைவு (Loss of Appetite)"),
    "itching": st.checkbox("இரைப்பு (Itching)"),
    "skin_rash": st.checkbox("தோல் படிந்து விடுதல் (Skin Rash)"),
    "joint_pain": st.checkbox("மூட்டு வலி (Joint Pain)"),
    "weight_loss": st.checkbox("எடை குறைவு (Weight Loss)"),
    "diarrhoea": st.checkbox("வயிற்றுப்போக்கு (Diarrhoea)"),
    "sweating": st.checkbox("வியர்வை அதிகம் (Sweating)"),
    "anxiety": st.checkbox("பதட்டம் (Anxiety)"),
    "blurred_and_distorted_vision": st.checkbox("பார்வை மங்கல் (Blurred Vision)"),
    "dizziness": st.checkbox("தலையிளக்கம் (Dizziness)"),
    "redness_of_eyes": st.checkbox("கண்களில் சிவப்பு (Redness of Eyes)")
}

# Input array உருவாக்க
input_data = np.array([1 if val else 0 for val in input_features.values()]).reshape(1, -1)

# Predict button
if st.button("நோயை கணிக்கவும் (Predict Disease)"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"கணிக்கப்பட்ட நோய் வகை: {prediction}")
    except Exception as e:
        st.error(f"பிழை: {e}")
