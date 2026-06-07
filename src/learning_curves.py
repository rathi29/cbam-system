import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
import xgboost as xgb
from src.database import get_all_shipments
from src.utils import generate_mock_shipments

def plot_learning_curve_data():
    """Generate learning curve data"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 10:
        shipments = generate_mock_shipments(30)
    
    df = pd.DataFrame(shipments)
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    y = df['direct_emissions']
    
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'], drop_first=False)
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=42)
    
    # Generate learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_encoded, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='r2'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    return {
        'train_sizes': train_sizes.tolist(),
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist(),
        'val_mean': val_mean.tolist(),
        'val_std': val_std.tolist()
    }