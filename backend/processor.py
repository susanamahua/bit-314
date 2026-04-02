import pandas as pd
import numpy as np

def calculate_spi(df, window_size=3):
    """
    Calculates a proxy Standardized Precipitation Index (SPI) using a rolling window.
    """
    # Requires a larger historical baseline for a real SPI, but for a 5-yr mock we use
    # a simple standardized Z-score over a rolling window per county.
    df['rolling_precip'] = df.groupby('county')['precipitation_mm'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    df['mean_precip'] = df.groupby('county')['rolling_precip'].transform('mean')
    df['std_precip'] = df.groupby('county')['rolling_precip'].transform('std')
    
    # SPI approximation
    df['spi'] = (df['rolling_precip'] - df['mean_precip']) / (df['std_precip'] + 1e-6)
    return df

def feature_engineering(df):
    """
    Transforms raw satellite bands into meaningful drought/scarcity indices.
    """
    print("Performing feature engineering on satellite data...")
    # Calculate SPI
    df = calculate_spi(df)
    
    # Calculate VHI (Vegetation Health Index) rough proxy
    # Highly correlated with NDVI and inversely with Temperature
    # normalize variables
    df['norm_ndvi'] = (df['ndvi'] - df['ndvi'].min()) / (df['ndvi'].max() - df['ndvi'].min() + 1e-6)
    df['norm_temp'] = (df['temperature_c'] - df['temperature_c'].min()) / (df['temperature_c'].max() - df['temperature_c'].min() + 1e-6)
    
    # VHI proxy (0 to 1, where 0 is extreme stress, 1 is excellent health)
    df['vhi'] = (0.5 * df['norm_ndvi']) + (0.5 * (1 - df['norm_temp']))
    
    # Define Target: "Water Scarcity Index (WSI)" or Hotspot Probability label.
    # WSI proxy calculation (Higher means more scarcity)
    # Drought happens if SPI is low, soil moisture is low, and VHI is low
    df['wsi_score'] = (1 - df['spi']) * 0.4 + (1 - df['soil_moisture']) * 0.3 + (1 - df['vhi']) * 0.3
    
    # Convert score to a binary label (1 = Hotspot, 0 = Normal) 
    # Let's say top 20% of scores indicate a hotspot
    threshold = df['wsi_score'].quantile(0.8)
    df['is_hotspot'] = (df['wsi_score'] >= threshold).astype(int)
    
    # Lagged features for prediction (predicting next month's hotspot)
    df = df.sort_values(by=['county', 'date'])
    df['spi_lag1'] = df.groupby('county')['spi'].shift(1)
    df['vhi_lag1'] = df.groupby('county')['vhi'].shift(1)
    df['sm_lag1'] = df.groupby('county')['soil_moisture'].shift(1)
    
    # Drop NaNs from lagging
    df = df.dropna().reset_index(drop=True)
    return df
