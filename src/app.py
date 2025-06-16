import streamlit as st
import joblib
import numpy as np
import os

modelo = joblib.load("modelo_personalidad.pkl")

st.title("Personality Prediction Form")

time_spent_alone = st.slider("Hours spent alone daily:", min_value=0.0, max_value=11.0, value=5.0, step=0.5)
stage_fear = st.selectbox("Stage Fear:?", ["Yes", "No"])
social_event_attendance = st.slider("Frequency of social events:", min_value=0.0, max_value=10.0, value=5.0, step=1.0)
going_outside = st.slider("Frequency of going outside:", min_value=0.0, max_value=7.0, value=3.0, step=1.0)
drained_after = st.selectbox("Feeling drained after socializing:", ["Yes", "No"])
friends_circle_size = st.slider("Number of close friends:", min_value=0.0, max_value=15.0, value=7.0, step=1.0)
post_frequency = st.slider("Social media post frequency (0–10):", min_value=0.0, max_value=10.0, value=5.0, step=1.0)

stage_fear_bin = 1 if stage_fear == "Yes" else 0
drained_bin = 1 if drained_after == "Yes" else 0

if st.button("Predecir personalidad"):
    entrada = np.array([[time_spent_alone, stage_fear_bin, social_event_attendance,
                         going_outside, drained_bin, friends_circle_size, post_frequency]])
    
    resultado = modelo.predict(entrada)[0]
    personalidad = "Introvert" if resultado == 1 else "Extrovert"

    st.success(f"La personalidad predicha es: **{personalidad}**")