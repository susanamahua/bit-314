import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from .config import DATA_DIR, MODEL_DIR
from .data_fetcher import generate_mock_historical_data
from .processor import feature_engineering

# Ensure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_data(data_path):
    if not os.path.exists(data_path):
        df = generate_mock_historical_data(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Process features
    processed_df = feature_engineering(df)
    
    # We predict whether the NEXT month is a hotspot (is_hotspot) using lag features
    features = ['spi_lag1', 'vhi_lag1', 'sm_lag1', 'precipitation_mm', 'temperature_c']
    target = 'is_hotspot'
    
    X = processed_df[features]
    y = processed_df[target]
    
    # Additionally keep info for inference mapping later
    meta = processed_df[['county', 'date', 'is_hotspot']]
    
    return X, y, meta, processed_df

def train_model():
    print("Training Water Scarcity Prediction Model...")
    data_path = os.path.join(DATA_DIR, "historical_data.csv")
    X, y, meta, full_df = prepare_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model trained. Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    
    model_path = os.path.join(MODEL_DIR, "xgb_hotspot_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save training dataset to make inference easy on stream lit
    full_df.to_csv(os.path.join(DATA_DIR, "processed_scarcity_data.csv"), index=False)
    
if __name__ == "__main__":
    train_model()
