import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import joblib

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import SELECTED_COUNTIES, DATA_DIR, MODEL_DIR

st.set_page_config(page_title="Kenya Water Scarcity Prediction v2", layout="wide")

@st.cache_data
def load_data():
    dl_path = os.path.join(DATA_DIR, "processed_dl_data.csv")
    if os.path.exists(dl_path):
        return pd.read_csv(dl_path), True
        
    file_path = os.path.join(DATA_DIR, "processed_scarcity_data.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path), False
    return pd.DataFrame(), False

# Check what Data is available
df, is_dl = load_data()

st.title("Kenya Water Scarcity Prediction Platform (V2)")
if is_dl:
    st.markdown("⚠️ **Deep Learning Neural Network Mode Active**")
else:
    st.markdown("Predicting regions at high risk for water scarcity based on satellite imagery and climate data.")

if df.empty:
    st.warning("Data not found. Please run the backend training pipelines first.")
else:
    st.sidebar.header("Filter Options")
    
    selected_time = st.sidebar.selectbox("Select Prediction Date", df['date'].unique(), index=len(df['date'].unique())-1)
    
    current_df = df[df['date'] == selected_time].copy()
    
    if is_dl:
        try:
            scaler = joblib.load(os.path.join(MODEL_DIR, "dl_scaler.pkl"))
            dl_model = joblib.load(os.path.join(MODEL_DIR, "dl_net_model.pkl"))
            
            features = ['spi_lag1', 'vhi_lag1', 'sm_lag1', 'precipitation_mm', 'temperature_c']
            X_scaled = scaler.transform(current_df[features].values)
            current_df['Predict_Hotspot'] = dl_model.predict(X_scaled)
            
        except Exception as e:
            st.error(f"Failed to load the Neural Network: {e}")
            current_df['Predict_Hotspot'] = 0
            
    else:
        model_path = os.path.join(MODEL_DIR, "xgb_hotspot_model.pkl")
        if os.path.exists(model_path):
            xgb_model = joblib.load(model_path)
            features = ['spi_lag1', 'vhi_lag1', 'sm_lag1', 'precipitation_mm', 'temperature_c']
            current_df['Predict_Hotspot'] = xgb_model.predict(current_df[features])
        else:
            current_df['Predict_Hotspot'] = 0

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Drought Risk & Vulnerability Map - {selected_time}")
        m = folium.Map(location=[0.0236, 37.9062], zoom_start=6)
        
        mock_coords = {
            "Turkana": [3.1118, 35.6186], "Marsabit": [2.3308, 37.9942], "Nairobi": [-1.2921, 36.8219],
            "Wajir": [1.7483, 40.0558], "Mandera": [3.9356, 41.8596], "Garissa": [-0.4532, 39.6461],
            "Isiolo": [0.3546, 38.4842], "Samburu": [1.2185, 36.8157], "Laikipia": [0.3606, 36.9930],
            "Meru": [0.0463, 37.6559], "Tharaka Nithi": [-0.2974, 38.0062], "Embu": [-0.5312, 37.4506],
            "Kitui": [-1.3683, 37.9942], "Machakos": [-1.5177, 37.2634], "Makueni": [-2.2530, 37.6329],
            "Nyandarua": [-0.1804, 36.3776], "Nyeri": [-0.4194, 36.9536], "Kirinyaga": [-0.5055, 37.2831],
            "Murang'a": [-0.7167, 37.1472], "Kiambu": [-1.1714, 36.8356]
        }
        
        for idx, row in current_df.iterrows():
            c_name = row['county']
            if c_name in mock_coords:
                is_risk = row['Predict_Hotspot'] == 1
                color = 'red' if is_risk else 'green'
                folium.CircleMarker(
                    location=mock_coords[c_name],
                    radius=10 + (2 if is_dl else 0), # Slightly larger for DL
                    popup=f"{c_name}: {'High Vulnerability Risk' if is_risk else 'Stable'}",
                    color=color,
                    fill=True,
                    fill_color=color
                ).add_to(m)
                
        st_folium(m, width=700, height=500)
        
    with col2:
        st.subheader("Crisis Intervention Zone")
        at_risk = current_df[current_df['Predict_Hotspot'] == 1]['county'].tolist()
        if at_risk:
            st.error(f"{len(at_risk)} zones hitting critical vulnerability markers.")
            for county in at_risk:
                st.write(f"⚠️ {county}")
            if st.button("Manually Dispatch SMS Alerts - Africa's Talking"):
                from backend.alerter import send_alert_sms
                res = send_alert_sms(["+254_STAKEHOLDERS"], f"MANUAL OVERRIDE: {len(at_risk)} zones critically vulnerable.")
                st.success("Alert Broadcast Processed!")
                st.json(res)
        else:
            st.success("No high-risk zones detected by the model.")
