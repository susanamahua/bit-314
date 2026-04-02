import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from .config import DATA_DIR, MODEL_DIR
from .data_fetcher import generate_mock_historical_data
from .processor import feature_engineering
from .alerter import send_alert_sms

os.makedirs(MODEL_DIR, exist_ok=True)

def train_dl_model():
    print("Training Deep Learning Neural Network (MLP Surrogate due to PyTorch/Python3.14 limits)...")
    data_path = os.path.join(DATA_DIR, "historical_data.csv")
    if not os.path.exists(data_path):
        df = generate_mock_historical_data(data_path)
    else:
        df = pd.read_csv(data_path)
        
    processed_df = feature_engineering(df)
    
    features = ['spi_lag1', 'vhi_lag1', 'sm_lag1', 'precipitation_mm', 'temperature_c']
    target_col = 'is_hotspot'
    
    X = processed_df[features].values
    y = processed_df[target_col].values
    
    # Scale features for the Neural Net
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dump the scaler for inference
    joblib.dump(scaler, os.path.join(MODEL_DIR, "dl_scaler.pkl"))
    
    # Train MLP Classifier (Deep Neural Network approach)
    # Using 3 hidden layers (64, 32, 16)
    model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42)
    model.fit(X_scaled, y)
    
    model_path = os.path.join(MODEL_DIR, "dl_net_model.pkl")
    joblib.dump(model, model_path)
    print(f"Deep Neural Network successfully saved to {model_path}")
    
    # Save the processed records back for frontend
    processed_df.to_csv(os.path.join(DATA_DIR, "processed_dl_data.csv"), index=False)
    
    # Dispatch an alert if any county tripped the threshold on the last active date
    latest_date = processed_df['date'].max()
    latest_data = processed_df[processed_df['date'] == latest_date]
    at_risk = latest_data[latest_data['is_hotspot'] == 1]['county'].tolist()
    
    if at_risk:
        send_alert_sms(["+254_STAKEHOLDERS"], f"Deep Learning Model flags {len(at_risk)} counties at high risk for the upcoming month: {', '.join(at_risk[:3])}...")
        
if __name__ == "__main__":
    train_dl_model()
