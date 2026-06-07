import pickle
import pandas as pd
import xgboost as xgb
import json
from datetime import datetime
from src.database import get_all_shipments
from src.utils import generate_mock_shipments

def train_cbam_model():
    shipments = get_all_shipments()
    
    if len(shipments) < 5:
        print("⚠️ Using synthetic data...")
        shipments = generate_mock_shipments(20)
    
    df = pd.DataFrame(shipments)
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    y = df['direct_emissions']
    
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'], drop_first=False)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_encoded, y)
    
    r2 = model.score(X_encoded, y)
    
    with open('models/cbam_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_names': X_encoded.columns.tolist()}, f)
    
    version_info = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'XGBoost',
        'r2_score': float(r2),
        'records_used': len(shipments),
        'features': X_encoded.columns.tolist()
    }
    
    with open('models/model_history.json', 'a') as f:
        f.write(json.dumps(version_info) + '\n')
    
    print(f"✅ Model trained. R² = {r2:.3f}")
    return r2

def predict_tax(material_type, tonnage, country):
    try:
        with open('models/cbam_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        return {'error': 'Model not trained yet'}
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    input_df = pd.DataFrame({
        'material_type': [material_type],
        'tonnage': [tonnage],
        'country_of_origin': [country]
    })
    
    input_encoded = pd.get_dummies(input_df, columns=['material_type', 'country_of_origin'])
    
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_names]
    
    predicted_emissions = model.predict(input_encoded)[0]
    eu_cbam_rate = 85
    predicted_tax = predicted_emissions * tonnage * eu_cbam_rate
    
    return {
        'predicted_emissions': round(predicted_emissions, 3),
        'predicted_tax_eur': round(predicted_tax, 2),
        'tonnage': tonnage,
        'material': material_type,
        'country': country
    }