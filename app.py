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

# Input fields
temperature = st.sidebar.slider("Temperature (°C)", 0, 60, 25)
charging_cycles = st.sidebar.number_input("Charging Cycles", 0, 5000, 800)
voltage = st.sidebar.slider("Voltage (V)", 2.5, 4.5, 3.8)
current = st.sidebar.slider("Current (A)", -200, 200, 15)
soc = st.sidebar.slider("State of Charge (%)", 0, 100, 65)

if st.button("🔮 Predict Battery Health", type="primary"):
    
    # IMPORTANT: Match exact column names and order used during training
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'charging_cycles': [charging_cycles],
        'voltage': [voltage],
        'current': [current],
        'soc': [soc]
    })
    
    # Transform using scaler
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"**Predicted Battery Health: {prediction:.2f}%**")
    
    if prediction >= 85:
        st.success("🟢 Excellent Condition - Battery is performing very well")
    elif prediction >= 70:
        st.warning("🟡 Good Condition - Minor degradation")
    elif prediction >= 50:
        st.warning("🟠 Fair Condition")
    else:
        st.error("🔴 Poor Condition - Needs attention")

st.caption("EV Battery SoH Prediction | Sagar | Bengaluru")
