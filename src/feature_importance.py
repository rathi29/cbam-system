import pandas as pd
import xgboost as xgb
from src.database import get_all_shipments
from src.utils import generate_mock_shipments

def get_feature_importance():
    """Get feature importance from trained model"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 5:
        shipments = generate_mock_shipments(20)
    
    df = pd.DataFrame(shipments)
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    y = df['direct_emissions']
    
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'], drop_first=False)
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_encoded, y)
    
    # Get importances
    importances = model.feature_importances_
    feature_names = X_encoded.columns.tolist()
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    return importance_df