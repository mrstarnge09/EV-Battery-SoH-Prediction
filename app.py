
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('ev_battery_soh_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="EV Battery Predictor", page_icon="⚡")

st.title("⚡ EV Battery State of Health Predictor")
st.markdown("### Predict the remaining health of an Electric Vehicle Battery")

st.sidebar.header("🔧 Enter Battery Parameters")

# Input fields - aligned with training data features
total_distance_km = st.sidebar.number_input("Total Distance (km)", min_value=1000.0, max_value=40000.0, value=20000.0, step=100.0)
average_trip_speed_kmph = st.sidebar.slider("Average Trip Speed (km/h)", min_value=20.0, max_value=60.0, value=40.0, step=0.1)
ambient_temperature_C = st.sidebar.slider("Ambient Temperature (°C)", min_value=10.0, max_value=45.0, value=27.5, step=0.1)
trip_duration_min = st.sidebar.slider("Trip Duration (min)", min_value=10.0, max_value=90.0, value=50.0, step=0.1)
charging_cycles = st.sidebar.number_input("Charging Cycles", min_value=50, max_value=800, value=370)
fast_charging_ratio_percent = st.sidebar.slider("Fast Charging Ratio (%)", min_value=0.0, max_value=90.0, value=28.7, step=0.1)
average_battery_temperature_C = st.sidebar.slider("Average Battery Temperature (°C)", min_value=25.0, max_value=50.0, value=35.3, step=0.1)


if st.button("🔮 Predict Battery Health"):
    input_data = pd.DataFrame({
        'total_distance_km': [total_distance_km],
        'average_trip_speed_kmph': [average_trip_speed_kmph],
        'ambient_temperature_C': [ambient_temperature_C],
        'trip_duration_min': [trip_duration_min],
        'charging_cycles': [charging_cycles],
        'fast_charging_ratio_%': [fast_charging_ratio_percent],
        'average_battery_temperature_C': [average_battery_temperature_C]
    })

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"**Predicted Battery Health: {prediction:.2f}%**")

    if prediction >= 85:
        st.success("🟢 Excellent Condition")
    elif prediction >= 70:
        st.warning("🟡 Good Condition")
    else:
        st.error("🔴 Poor Condition")
