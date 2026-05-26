import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('ev_battery_soh_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="EV Battery Predictor", page_icon="⚡")

st.title("⚡ EV Battery State of Health Predictor")
st.markdown("### Predict Remaining Battery Health")

st.sidebar.header("🔧 Input Battery Parameters")

# Updated inputs matching your model's features
total_distance = st.sidebar.number_input("Total Distance (km)", 0, 10000, 500)
avg_speed = st.sidebar.slider("Average Trip Speed (km/h)", 0, 120, 45)
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", -10, 50, 28)
trip_duration = st.sidebar.number_input("Trip Duration (minutes)", 0, 300, 45)
charging_cycles = st.sidebar.number_input("Charging Cycles", 0, 5000, 800)
fast_charging_ratio = st.sidebar.slider("Fast Charging Ratio (%)", 0, 100, 40)
battery_temp = st.sidebar.slider("Average Battery Temperature (°C)", 0, 60, 35)

if st.button("🔮 Predict Battery Health", type="primary"):
    
    input_data = pd.DataFrame({
        'total_distance_km': [total_distance],
        'average_trip_speed_kmph': [avg_speed],
        'ambient_temperature_C': [ambient_temp],
        'trip_duration_min': [trip_duration],
        'charging_cycles': [charging_cycles],
        'fast_charging_ratio_%': [fast_charging_ratio],
        'average_battery_temperature_C': [battery_temp]
    })
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"**Predicted Battery Health: {prediction:.2f}%**")
    
    if prediction >= 85:
        st.success("🟢 Excellent Condition")
    elif prediction >= 70:
        st.warning("🟡 Good Condition")
    elif prediction >= 50:
        st.warning("🟠 Fair Condition")
    else:
        st.error("🔴 Poor Condition")

st.caption("EV Battery SoH Prediction | Sagar | Bengaluru")
