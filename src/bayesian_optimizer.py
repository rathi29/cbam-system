import pickle
import pandas as pd
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Integer, Real
from src.database import get_all_shipments
from src.utils import generate_mock_shipments

def optimize_model_hyperparameters():
    """Use Bayesian optimization to find best hyperparameters"""
    
    shipments = get_all_shipments()
    
    if len(shipments) < 5:
        shipments = generate_mock_shipments(20)
    
    df = pd.DataFrame(shipments)
    
    X = df[['material_type', 'tonnage', 'country_of_origin']].copy()
    y = df['direct_emissions']
    
    X_encoded = pd.get_dummies(X, columns=['material_type', 'country_of_origin'], drop_first=False)
    
    # Define objective function (minimize = negative R²)
    def objective(params):
        max_depth, n_estimators, learning_rate = params
        
        model = xgb.XGBRegressor(
            max_depth=int(max_depth),
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            random_state=42
        )
        
        # Cross-validation score
        scores = cross_val_score(model, X_encoded, y, cv=5, scoring='r2')
        return -scores.mean()  # Negative because we minimize
    
    # Define search space
    space = [
        Integer(3, 20, name='max_depth'),
        Integer(50, 300, name='n_estimators'),
        Real(0.01, 0.3, name='learning_rate')
    ]
    
    print("⏳ Optimizing hyperparameters (this takes 2-3 min)...")
    
    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=25,  # Test 25 combinations
        n_initial_points=5,
        random_state=42,
        verbose=1
    )
    
    best_params = {
        'max_depth': int(result.x[0]),
        'n_estimators': int(result.x[1]),
        'learning_rate': round(result.x[2], 3),
        'best_score': round(-result.fun, 3)
    }
    
    print(f"✅ Best params: {best_params}")
    
    # Train final model with best params
    model = xgb.XGBRegressor(
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        random_state=42
    )
    
    model.fit(X_encoded, y)
    
    # Save
    with open('models/cbam_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_names': X_encoded.columns.tolist()}, f)
    
    return best_params