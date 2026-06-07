import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from src.database import get_all_shipments
from src.utils import generate_mock_shipments

def compare_models():
    """Compare 3 models"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 5:
        shipments = generate_mock_shipments(20)
    
    df = pd.DataFrame(shipments)
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    y = df['direct_emissions']
    
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'], drop_first=False)
    
    results = {}
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=42)
    xgb_score = cross_val_score(xgb_model, X_encoded, y, cv=5, scoring='r2').mean()
    results['XGBoost'] = round(xgb_score, 4)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_score = cross_val_score(rf_model, X_encoded, y, cv=5, scoring='r2').mean()
    results['Random Forest'] = round(rf_score, 4)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_score = cross_val_score(lr_model, X_encoded, y, cv=5, scoring='r2').mean()
    results['Linear Regression'] = round(lr_score, 4)
    
    return results