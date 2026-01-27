# mppt_kaggle_plant1.py (Updated for PyTorch)
# -*- coding: utf-8 -*-
"""
Solar Panel MPPT Prediction using ANN (Kaggle Plant 1 Data)
Inputs: Irradiance, Temperature; Outputs: V_mp, P_mp
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import os

# Config
GEN_FILE = "data/Plant_1_Generation_Data.csv"
WEA_FILE = "data/Plant_1_Weather_Sensor_Data.csv"
OUT_PREFIX = "plant1"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_and_preprocess():
    """Load, merge, and preprocess Kaggle Plant 1 data."""
    gen = pd.read_csv(GEN_FILE, low_memory=False)
    wea = pd.read_csv(WEA_FILE, low_memory=False)

    # Find timestamp columns
    def guess_col(df, keywords):
        for kw in keywords:
            for col in df.columns:
                if kw.lower() in col.lower():
                    return col
        return None

    t1 = guess_col(gen, ['date', 'time'])
    t2 = guess_col(wea, ['date', 'time'])
    if t1 and t2:
        gen[t1] = pd.to_datetime(gen[t1], errors='coerce')
        wea[t2] = pd.to_datetime(wea[t2], errors='coerce')
        gen = gen.set_index(t1).sort_index()
        wea = wea.set_index(t2).sort_index()
        df = gen.join(wea, how='inner', lsuffix='_gen', rsuffix='_wea')
    else:
        df = pd.concat([gen, wea], axis=1)

    # Extract columns (hardcoding for reliability based on Kaggle dataset)
    # From inspection: irradiation -> 'IRRADIATION', temp -> 'MODULE_TEMPERATURE' or 'AMBIENT_TEMPERATURE', power -> 'DC_POWER', no direct V_mp
    irr_col = 'IRRADIATION'
    temp_col = 'MODULE_TEMPERATURE'  # Using module temp as it's more relevant for panel performance
    power_col = 'DC_POWER'
    volt_col = None  # No voltage in dataset, will estimate

    # Create DataFrame
    dfm = pd.DataFrame({
        'Irradiance': pd.to_numeric(df[irr_col], errors='coerce') if irr_col in df else np.nan,
        'Temperature': pd.to_numeric(df[temp_col], errors='coerce') if temp_col in df else np.nan,
        'Power': pd.to_numeric(df[power_col], errors='coerce') if power_col in df else np.nan,
        'V_mp': np.nan  # Will estimate
    })

    # Handle missing V_mp, assume typical V_mp adjustment for temperature (simplified model)
    if dfm['V_mp'].isna().all():
        dfm['V_mp'] = 30.0 * (1 - 0.003 * (dfm['Temperature'] - 25.0))  # Example temp coefficient
    dfm['P_mp'] = dfm['Power']

    # Clean: interpolate, filter, drop NaNs
    dfm = dfm.sort_index().interpolate(method='time').dropna()
    dfm = dfm[(dfm['Irradiance'] >= 0) & (dfm['Power'] >= 0) & (dfm['V_mp'] >= 0)]

    return dfm

def train_evaluate_plot(dfm):
    """Train ANN with PyTorch, evaluate, plot results, and save model/scalers."""
    # Prepare data
    X = dfm[['Irradiance', 'Temperature']].values
    y = dfm[['V_mp', 'P_mp']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Scale
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train_s = X_scaler.transform(X_train)
    X_test_s = X_scaler.transform(X_test)
    y_train_s = y_scaler.transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_s, dtype=torch.float32)

    # DataLoader for batching
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define model
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

    model = MPPTModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 150
    best_loss = float('inf')
    patience = 8
    counter = 0
    history = {'loss': [], 'mae': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            train_mae += torch.sum(torch.abs(outputs - y_batch)).item()

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        history['loss'].append(train_loss)
        history['mae'].append(train_mae)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()
            val_mae = torch.mean(torch.abs(val_outputs - y_test_t)).item()
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Early stopping and checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{OUT_PREFIX}_best.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # Load best model
    model.load_state_dict(torch.load(f'{OUT_PREFIX}_best.pt'))

    # Evaluate
    with torch.no_grad():
        y_pred_s = model(X_test_t).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_s)
    print(f"V_mp: RMSE={np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0])):.3f} V, "
          f"MAE={mean_absolute_error(y_test[:, 0], y_pred[:, 0]):.3f} V, "
          f"R²={r2_score(y_test[:, 0], y_pred[:, 0]):.3f}")
    print(f"P_mp: RMSE={np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1])):.3f} W, "
          f"MAE={mean_absolute_error(y_test[:, 1], y_pred[:, 1]):.3f} W, "
          f"R²={r2_score(y_test[:, 1], y_pred[:, 1]):.3f}")

    # Save model/scalers
    joblib.dump(X_scaler, f'{OUT_PREFIX}_Xscaler.joblib')
    joblib.dump(y_scaler, f'{OUT_PREFIX}_yscaler.joblib')
    torch.save(model.state_dict(), f'{OUT_PREFIX}_model.pt')

    # Plot history
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], 'b-', label='Train')
    plt.plot(history['val_loss'], 'r-', label='Val')
    plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], 'b-', label='Train MAE')
    plt.plot(history['val_mae'], 'r-', label='Val MAE')
    plt.title('MAE'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_PREFIX}_history.png')
    plt.show()

    # Plot 3D surface (P_mp vs. Temperature, Irradiance)
    G_grid = np.linspace(dfm['Irradiance'].quantile(0.01), dfm['Irradiance'].quantile(0.99), 40)
    T_grid = np.linspace(dfm['Temperature'].quantile(0.01), dfm['Temperature'].quantile(0.99), 40)
    Gg, Tg = np.meshgrid(G_grid, T_grid)
    X_grid = np.column_stack([Gg.ravel(), Tg.ravel()])
    X_grid_s = X_scaler.transform(X_grid)
    X_grid_t = torch.tensor(X_grid_s, dtype=torch.float32)
    with torch.no_grad():
        P_pred_s = model(X_grid_t)[:, 1].numpy()
    P_pred = y_scaler.inverse_transform(np.column_stack([np.zeros_like(P_pred_s), P_pred_s]))[:, 1].reshape(Gg.shape)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Tg, Gg, P_pred, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Temp (°C)'); ax.set_ylabel('Irradiance (W/m²)'); ax.set_zlabel('P_mp (W)')
    ax.set_title('Predicted MPPT Power')
    plt.savefig(f'{OUT_PREFIX}_mppt_surface.png')
    plt.show()

    return model, X_scaler, y_scaler

def predict_new(model, X_scaler, y_scaler, irradiance, temperature):
    """Predict V_mp, P_mp for new inputs."""
    model.eval()
    X_new = np.array([[irradiance, temperature]])
    X_scaled = torch.tensor(X_scaler.transform(X_new), dtype=torch.float32)
    with torch.no_grad():
        y_pred_s = model(X_scaled).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_s)
    return {'V_mp': y_pred[0, 0], 'P_mp': y_pred[0, 1]}

# Run pipeline
dfm = load_and_preprocess()
model, X_scaler, y_scaler = train_evaluate_plot(dfm)
pred = predict_new(model, X_scaler, y_scaler, 800, 25)
print(f"Sample Prediction (G=800 W/m², T=25°C): V_mp={pred['V_mp']:.2f} V, P_mp={pred['P_mp']:.2f} W")