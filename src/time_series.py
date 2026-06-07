import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.database import get_all_shipments

def prepare_time_series_data(df, lookback=7):
    """Create lagged features for time series"""
    
    # Sort by date
    df = df.sort_values('created_at')
    
    # Create lags (past 7 days)
    df['emissions_lag_1'] = df['direct_emissions'].shift(1)
    df['emissions_lag_7'] = df['direct_emissions'].shift(lookback)
    df['emissions_ma_7'] = df['direct_emissions'].rolling(window=7).mean()
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def train_time_series_model():
    """Train model to predict next period emissions"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 15:
        print("⚠️ Need 15+ records for time series")
        return None
    
    df = pd.DataFrame(shipments)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Prepare data with lags
    df = prepare_time_series_data(df)
    
    # Features: past emissions + metadata
    X = df[['emissions_lag_1', 'emissions_lag_7', 'emissions_ma_7', 'tonnage']].copy()
    y = df['direct_emissions']
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    
    score = model.score(X, y)
    print(f"✅ Time series model R² = {score:.3f}")
    
    return model

def forecast_emissions(days=30):
    """Forecast emissions for next N days"""
    
    shipments = get_all_shipments()
    df = pd.DataFrame(shipments)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = prepare_time_series_data(df)
    
    if len(df) < 10:
        return None
    
    # Train model
    X = df[['emissions_lag_1', 'emissions_lag_7', 'emissions_ma_7', 'tonnage']].copy()
    y = df['direct_emissions']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Generate forecast
    last_row = df.iloc[-1]
    forecast_values = []
    
    current_lag_1 = last_row['direct_emissions']
    current_lag_7 = last_row['emissions_lag_7']
    current_ma = last_row['emissions_ma_7']
    avg_tonnage = df['tonnage'].mean()
    
    for i in range(days):
        input_data = np.array([[current_lag_1, current_lag_7, current_ma, avg_tonnage]])
        pred = model.predict(input_data)[0]
        forecast_values.append(round(pred, 3))
        
        # Update for next iteration
        current_lag_1 = pred
        current_lag_7 = current_lag_7 if i < 7 else forecast_values[i-7]
    
    return forecast_values