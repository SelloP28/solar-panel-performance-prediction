# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the saved model and scalers (generated from mppt_kaggle_plant1.py)
@st.cache_resource
def load_model_and_scalers():
    model = keras.models.load_model('plant1_model.h5', compile=False)
    X_scaler = joblib.load('plant1_Xscaler.joblib')
    y_scaler = joblib.load('plant1_yscaler.joblib')
    return model, X_scaler, y_scaler

model, X_scaler, y_scaler = load_model_and_scalers()

# App title and description (based on your CV project details)
st.title("Solar Panel Performance Prediction")
st.markdown("""
This app predicts Maximum Power Point Tracking (MPPT) voltage (V_mp) and power (P_mp) for solar panels 
using an Artificial Neural Network (ANN) model trained on historical data. 
Enter irradiance and temperature values below to get predictions.
""")

# User inputs via sliders (reasonable ranges based on typical solar data)
irradiance = st.slider("Irradiance (W/m²)", min_value=0.0, max_value=1500.0, value=800.0, step=10.0)
temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=1.0)

# Prediction function (adapted from predict_new in your script)
def predict(irradiance, temperature):
    X_new = np.array([[irradiance, temperature]])
    X_scaled = X_scaler.transform(X_new)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return y_pred[0][0], y_pred[0][1]  # V_mp, P_mp

# Button to trigger prediction
if st.button("Predict MPPT Values"):
    v_mp, p_mp = predict(irradiance, temperature)
    st.success(f"Predicted MPPT Voltage (V_mp): {v_mp:.2f} V")
    st.success(f"Predicted MPPT Power (P_mp): {p_mp:.2f} W")

# Optional: 3D visualization section (inspired by your script's surface plot)
st.subheader("Power Variation Visualization")
st.markdown("This 3D plot shows predicted P_mp across a range of temperatures and irradiances for insights.")

# Generate grid for 3D plot
T_grid = np.linspace(0, 60, 20)
G_grid = np.linspace(0, 1500, 20)
Gg, Tg = np.meshgrid(G_grid, T_grid)
X_grid = np.column_stack([Gg.ravel(), Tg.ravel()])
P_pred = y_scaler.inverse_transform(model.predict(X_scaler.transform(X_grid)))[:, 1].reshape(Gg.shape)

# Create and display 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Tg, Gg, P_pred, cmap='viridis')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Irradiance (W/m²)')
ax.set_zlabel('P_mp (W)')
ax.set_title('Predicted MPPT Power Surface')
st.pyplot(fig)

# Footer with your details from CV
st.markdown("---")
st.markdown("Built by Sello Phakoe | GitHub: [SelloP28](https://github.com/SelloP28) | Email: u13238940@tuks.co.za")