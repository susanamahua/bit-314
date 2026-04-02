import pandas as pd
import numpy as np
import datetime
import os
from .config import SELECTED_COUNTIES, START_YEAR, END_YEAR, DATA_DIR

def generate_mock_historical_data(file_path):
    """
    Generates mock historical data for fast prototyping and testing.
    This bypasses the strict GEE limits for 5-yr downloads.
    """
    print(f"Generating mock historical data for {START_YEAR} to {END_YEAR}...")
    
    dates = pd.date_range(start=f'{START_YEAR}-01-01', end=f'{END_YEAR}-12-31', freq='ME')
    data = []
    
    for county in SELECTED_COUNTIES:
        for date in dates:
            # Add some seasonality to mock data
            month = date.month
            is_rainy_season = month in [3, 4, 5, 10, 11]
            
            precip = np.random.uniform(50, 200) if is_rainy_season else np.random.uniform(0, 50)
            temp = np.random.uniform(22, 28) if is_rainy_season else np.random.uniform(26, 35)
            sm = np.random.uniform(0.2, 0.4) if is_rainy_season else np.random.uniform(0.05, 0.2)
            ndvi = np.random.uniform(0.4, 0.8) if is_rainy_season else np.random.uniform(0.1, 0.4)
            ndwi = np.random.uniform(0.0, 0.3) if is_rainy_season else np.random.uniform(-0.2, 0.0)
            
            data.append({
                'county': county,
                'date': date,
                'precipitation_mm': precip,
                'temperature_c': temp,
                'soil_moisture': sm,
                'ndvi': ndvi,
                'ndwi': ndwi
            })
            
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} records of historical data to {file_path}")
    return df

def fetch_gee_precipitation(start_date, end_date):
    """
    Example function placeholder to extract real CHIRPS data from Earth Engine.
    This requires ee to be authenticated.
    """
    # ... Uses ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") ...
    pass

if __name__ == "__main__":
    generate_mock_historical_data(os.path.join(DATA_DIR, "historical_data.csv"))
