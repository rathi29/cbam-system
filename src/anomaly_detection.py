import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.database import get_all_shipments

def detect_anomalies(contamination=0.15):
    """Find unusual shipments"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 10:
        return None
    
    df = pd.DataFrame(shipments)
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'], drop_first=False)
    
    # Train anomaly detector
    detector = IsolationForest(contamination=contamination, random_state=42)
    predictions = detector.fit_predict(X_encoded)
    
    # Get anomalies (prediction = -1)
    anomaly_indices = np.where(predictions == -1)[0]
    
    anomalies = []
    for idx in anomaly_indices:
        row = df.iloc[idx]
        anomalies.append({
            'shipment_id': row['shipment_id'],
            'material': row['material_type'],
            'tonnage': row['tonnage'],
            'country': row['country_of_origin'],
            'emissions': row['direct_emissions'],
            'flag': 'ANOMALY - Review'
        })
    
    return anomalies