import pickle
import pandas as pd
import numpy as np
from src.database import get_all_shipments

def get_uncertain_predictions(top_n=20):
    """Find shipments with unusual patterns"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 10:
        return None
    
    df = pd.DataFrame(shipments)
    
    try:
        with open('models/cbam_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        return None
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'])
    
    for col in feature_names:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    
    X_encoded = X_encoded[feature_names]
    
    # Get predictions
    predictions = model.predict(X_encoded)
    
    # Calculate uncertainty as deviation from average
    avg_emissions = np.mean(predictions)
    uncertainty = np.abs(predictions - avg_emissions) / (avg_emissions + 0.01)
    
    # Get top uncertain indices
    uncertain_indices = np.argsort(uncertainty)[-top_n:][::-1]
    
    result = []
    for idx in uncertain_indices:
        row = df.iloc[idx]
        result.append({
            'shipment_id': row['shipment_id'],
            'material_type': row['material_type'],
            'tonnage': row['tonnage'],
            'country_of_origin': row['country_of_origin'],
            'direct_emissions': row['direct_emissions'],
            'uncertainty_score': round(float(uncertainty[idx]), 3),
            'recommendation': 'AUDIT' if uncertainty[idx] > uncertainty.mean() else 'OK'
        })
    
    return result

def uncertainty_distribution():
    """Get overall uncertainty stats"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 10:
        return None
    
    df = pd.DataFrame(shipments)
    
    try:
        with open('models/cbam_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        return None
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'])
    
    for col in feature_names:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    
    X_encoded = X_encoded[feature_names]
    
    predictions = model.predict(X_encoded)
    avg_emissions = np.mean(predictions)
    uncertainty = np.abs(predictions - avg_emissions) / (avg_emissions + 0.01)
    
    return {
        'mean_uncertainty': round(float(uncertainty.mean()), 3),
        'max_uncertainty': round(float(uncertainty.max()), 3),
        'min_uncertainty': round(float(uncertainty.min()), 3),
        'total_shipments': len(df),
        'audit_recommended': int(sum(uncertainty > uncertainty.mean()))
    }