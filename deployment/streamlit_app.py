import streamlit as st
import requests

st.title("Hospital Shortage Predictor")

api_url = st.text_input("API URL", "http://localhost:8000/predict")
target = st.selectbox("Target", ["PrimaryCareShortageArea", "MentalHealthShortageArea"])
category = st.text_input("Category", "General")
bed_size = st.selectbox("LICENSED_BED_SIZE", ["1-49", "50-99", "100-149", "150-199", "200-299", "300-499", "500+"])
visits = st.number_input("Tot_ED_NmbVsts", min_value=0.0, value=10000.0)
stations = st.number_input("EDStations", min_value=0.0, value=10.0)

if st.button("Predict"):
    payload = {
        "target": target,
        "data": {
            "Category": category,
            "LICENSED_BED_SIZE": bed_size,
            "Tot_ED_NmbVsts": visits,
            "EDStations": stations,
        },
    }

    try:
        r = requests.post(api_url, json=payload, timeout=20)
        r.raise_for_status()
        st.success("Prediction successful")
        st.json(r.json())
    except Exception as e:
        st.error(f"Request failed: {e}")
