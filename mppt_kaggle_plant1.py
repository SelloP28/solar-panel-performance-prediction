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
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import os

# Config
GEN_FILE = "Plant_1_Generation_Data.csv"
WEA_FILE = "Plant_1_Weather_Sensor_Data.csv"
OUT_PREFIX = "plant1"
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

    # Extract columns
    irr_col = guess_col(df, ['irradi', 'ghi', 'radiat'])
    temp_col = guess_col(df, ['temp', 'ambient'])
    power_col = guess_col(df, ['dc_power', 'ac_power', 'power', 'yield'])
    volt_col = guess_col(df, ['vmp', 'voltage', 'module_voltage'])

    # Create DataFrame
    dfm = pd.DataFrame({
        'Irradiance': pd.to_numeric(df[irr_col], errors='coerce') if irr_col else np.nan,
        'Temperature': pd.to_numeric(df[temp_col], errors='coerce') if temp_col else np.nan,
        'Power': pd.to_numeric(df[power_col], errors='coerce') if power_col else np.nan,
        'V_mp': pd.to_numeric(df[volt_col], errors='coerce') if volt_col else np.nan
    })

    # Handle missing V_mp, assume Power is P_mp
    if dfm['V_mp'].isna().all():
        dfm['V_mp'] = 30.0 * (1 - 0.003 * (dfm['Temperature'] - 25.0))
    dfm['P_mp'] = dfm['Power']

    # Clean: interpolate, filter, drop NaNs
    dfm = dfm.sort_index().interpolate(method='time').dropna()
    dfm = dfm[(dfm['Irradiance'] >= 0) & (dfm['Power'] >= 0) & (dfm['V_mp'] >= 0)]

    return dfm

def train_evaluate_plot(dfm):
    """Train ANN, evaluate, plot results, and save model/scalers."""
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

    # Build and train model
    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train_s, y_train_s, validation_split=0.2, epochs=150, batch_size=64,
                       callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=8, restore_best_weights=True),
                                  keras.callbacks.ModelCheckpoint(f'{OUT_PREFIX}_best.h5', save_best_only=True)],
                       verbose=1)

    # Evaluate
    y_pred = y_scaler.inverse_transform(model.predict(X_test_s))
    print(f"V_mp: RMSE={np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0])):.3f} V, "
          f"MAE={mean_absolute_error(y_test[:, 0], y_pred[:, 0]):.3f} V, "
          f"R²={r2_score(y_test[:, 0], y_pred[:, 0]):.3f}")
    print(f"P_mp: RMSE={np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1])):.3f} W, "
          f"MAE={mean_absolute_error(y_test[:, 1], y_pred[:, 1]):.3f} W, "
          f"R²={r2_score(y_test[:, 1], y_pred[:, 1]):.3f}")

    # Save model/scalers
    joblib.dump(X_scaler, f'{OUT_PREFIX}_Xscaler.joblib')
    joblib.dump(y_scaler, f'{OUT_PREFIX}_yscaler.joblib')
    model.save(f'{OUT_PREFIX}_model.h5')

    # Plot history
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label='Train')
    plt.plot(history.history['val_loss'], 'r-', label='Val')
    plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], 'b-', label='Train MAE')
    plt.plot(history.history.get('val_mae', history.history.get('val_mean_absolute_error')), 'r-', label='Val MAE')
    plt.title('MAE'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_PREFIX}_history.png')
    plt.show()

    # Plot 3D surface (P_mp vs. Temperature, Irradiance)
    G_grid = np.linspace(dfm['Irradiance'].quantile(0.01), dfm['Irradiance'].quantile(0.99), 40)
    T_grid = np.linspace(dfm['Temperature'].quantile(0.01), dfm['Temperature'].quantile(0.99), 40)
    Gg, Tg = np.meshgrid(G_grid, T_grid)
    X_grid = np.column_stack([Gg.ravel(), Tg.ravel()])
    P_pred = y_scaler.inverse_transform(model.predict(X_scaler.transform(X_grid)))[:, 1].reshape(Gg.shape)

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
    X_new = np.array([[irradiance, temperature]])
    y_pred = y_scaler.inverse_transform(model.predict(X_scaler.transform(X_new)))
    return {'V_mp': y_pred[0, 0], 'P_mp': y_pred[0, 1]}

# Run pipeline
dfm = load_and_preprocess()
model, X_scaler, y_scaler = train_evaluate_plot(dfm)
pred = predict_new(model, X_scaler, y_scaler, 800, 25)
print(f"Sample Prediction (G=800 W/m², T=25°C): V_mp={pred['V_mp']:.2f} V, P_mp={pred['P_mp']:.2f} W")