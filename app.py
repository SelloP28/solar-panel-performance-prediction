# app.py (Updated for PyTorch and Interactive Plot)
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the model class (must match training script)
class MPPTModel(nn.Module):
    def __init__(self):
        super(MPPTModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load the saved model and scalers
@st.cache_resource
def load_model_and_scalers():
    model = MPPTModel()
    model.load_state_dict(torch.load('plant1_model.pt', map_location=torch.device('cpu')))
    model.eval()
    X_scaler = joblib.load('plant1_Xscaler.joblib')
    y_scaler = joblib.load('plant1_yscaler.joblib')
    return model, X_scaler, y_scaler

model, X_scaler, y_scaler = load_model_and_scalers()

# App title and description
st.title("Solar Panel Performance Prediction")
st.markdown("""
This app predicts Maximum Power Point Tracking (MPPT) voltage (V_mp) and power (P_mp) for solar panels 
using an Artificial Neural Network (ANN) model trained on historical data. 
Enter irradiance and temperature values below to get predictions.
""")

# User inputs via sliders
irradiance = st.slider("Irradiance (W/m²)", min_value=0.0, max_value=1500.0, value=800.0, step=10.0)
temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=1.0)

# Prediction function
def predict(irradiance, temperature):
    X_new = np.array([[irradiance, temperature]])
    X_scaled = torch.tensor(X_scaler.transform(X_new), dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled = model(X_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.numpy())
    v_mp, p_mp = y_pred[0][0], y_pred[0][1]
    # Enforce physics: clip negatives + low-irradiance handling
    if irradiance < 10:  # very low light
        v_mp = 0.0
        p_mp = 0.0
    else:
        v_mp = max(v_mp, 0.0)  # or max(v_mp, 15.0) for more realism
        p_mp = max(p_mp, 0.0)
    return v_mp, p_mp

# Button to trigger prediction and plot
if st.button("Predict MPPT Values and Update Plot"):
    v_mp, p_mp = predict(irradiance, temperature)
    st.success(f"Predicted MPPT Voltage (V_mp): {v_mp:.2f} V")
    st.success(f"Predicted MPPT Power (P_mp): {p_mp:.2f} W")

    # 3D visualization section
    st.subheader("Power Variation Visualization")
    st.markdown("This 3D plot shows predicted P_mp across a range of temperatures and irradiances. Your selected point is marked in red.")

    # Generate grid for 3D plot
    T_grid = np.linspace(0, 60, 20)
    G_grid = np.linspace(0, 1500, 20)
    Gg, Tg = np.meshgrid(G_grid, T_grid)
    X_grid = np.column_stack([Gg.ravel(), Tg.ravel()])
    X_grid_s = X_scaler.transform(X_grid)
    X_grid_t = torch.tensor(X_grid_s, dtype=torch.float32)
    with torch.no_grad():
        P_pred_s = model(X_grid_t)[:, 1].numpy()
    P_pred = y_scaler.inverse_transform(np.column_stack([np.zeros_like(P_pred_s), P_pred_s]))[:, 1].reshape(Gg.shape)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Tg, Gg, P_pred, cmap='viridis', alpha=0.8)
    
    # Mark the user-selected point
    ax.scatter([temperature], [irradiance], [p_mp], color='red', s=50, label='Your Prediction')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Irradiance (W/m²)')
    ax.set_zlabel('P_mp (W)')
    ax.set_title('Predicted MPPT Power Surface')
    ax.legend()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built by Sello Phakoe | GitHub: [SelloP28](https://github.com/SelloP28) | Email: u13238940@tuks.co.za")