import pandas as pd
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from src.database import get_all_shipments
from src.utils import generate_mock_shipments

def cross_validate_model():
    """Evaluate model with 5-fold cross-validation"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 5:
        shipments = generate_mock_shipments(20)
    
    df = pd.DataFrame(shipments)
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    y = df['direct_emissions']
    
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'], drop_first=False)
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=42)
    
    # 5-fold cross-validation
    scores = cross_val_score(model, X_encoded, y, cv=5, scoring='r2')
    
    return {
        'fold_scores': [round(s, 4) for s in scores],
        'mean': round(scores.mean(), 4),
        'std': round(scores.std(), 4),
        'min': round(scores.min(), 4),
        'max': round(scores.max(), 4)
    }