import streamlit as st
import pandas as pd
import joblib

# Load model with error handling
try:
    model = joblib.load('ev_battery_soh_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Model files not found. Please check deployment files.")
    st.stop()

st.set_page_config(page_title="EV Battery Predictor", page_icon="⚡")

st.title("⚡ EV Battery State of Health Predictor")

st.sidebar.header("Input Battery Parameters")

temperature = st.sidebar.slider("Temperature (°C)", 0, 60, 25)
charging_cycles = st.sidebar.number_input("Charging Cycles", 0, 5000, 800)
voltage = st.sidebar.slider("Voltage (V)", 2.5, 4.5, 3.8)
current = st.sidebar.slider("Current (A)", -200, 200, 15)
soc = st.sidebar.slider("State of Charge (%)", 0, 100, 65)

if st.button("Predict Battery Health"):
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'charging_cycles': [charging_cycles],
        'voltage': [voltage],
        'current': [current],
        'soc': [soc]
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
