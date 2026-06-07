import pickle
import shap
import pandas as pd

def explain_prediction(material_type, tonnage, country):
    """Explain why model made this prediction"""
    
    # Load model
    with open('models/cbam_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Create input
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
    
    # Get base prediction
    prediction = model.predict(input_encoded)[0]
    
    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_encoded)
    
    # Extract contributions
    base_value = explainer.expected_value
    contributions = {}
    
    for i, feature in enumerate(feature_names):
        contribution = shap_values[0][i]
        if abs(contribution) > 0.01:  # Only show significant ones
            contributions[feature] = round(contribution, 3)
    
    # Sort by magnitude
    contributions = dict(sorted(contributions.items(), 
        key=lambda x: abs(x[1]), reverse=True))
    
    return {
        'base_value': round(base_value, 3),
        'prediction': round(prediction, 3),
        'contributions': contributions,
        'shap_values': shap_values[0].tolist()
    }