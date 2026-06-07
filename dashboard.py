import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.database import init_db, get_all_shipments, insert_shipment
from src.ml_forecaster import train_cbam_model, predict_tax
from src.shap_explainer import explain_prediction
from src.utils import generate_mock_shipments
from src.model_comparison import compare_models
from src.cross_validation import cross_validate_model
from src.feature_importance import get_feature_importance
from src.anomaly_detection import detect_anomalies
from src.learning_curves import plot_learning_curve_data

st.set_page_config(page_title="Carbon Tax Calculator", layout="wide")

init_db()

st.title("💰 Carbon Tax Calculator")
st.markdown("Find out how much EU carbon tax you'll pay for your shipment")

page = st.sidebar.radio("What do you want to do?", [
    "💡 Quick Tax Prediction",
    "📋 View Past Shipments",
    "🔍 Understand Your Cost",
    "📊 Model Comparison",
    "✅ Cross-Validation",
    "⭐ Feature Importance",
    "🚨 Anomaly Detection",
    "📈 Learning Curves",
    "🧠 Train the AI",
    "📥 Load Test Data"
])

# PAGE 1: QUICK TAX PREDICTION
if page == "💡 Quick Tax Prediction":
    st.header("💡 Quick Tax Prediction")
    st.markdown("Tell me what you want to buy, I'll tell you the best country to buy from")
    
    col1, col2 = st.columns(2)
    
    with col1:
        material = st.selectbox(
            "What material?",
            ["Steel", "Cement", "Aluminum", "Fertilizer", "Chemicals"],
            help="Select the material you want to import"
        )
    
    with col2:
        tonnage = st.number_input(
            "How many tonnes?",
            min_value=10.0, max_value=500.0, value=100.0, step=10.0,
            help="Weight of shipment"
        )
    
    if st.button("🔍 Compare All Countries", key="compare", use_container_width=True):
        st.success("✅ Comparing all countries...")
        
        countries = ["China", "India", "Russia", "Turkey", "Ukraine"]
        results = []
        
        for country in countries:
            result = predict_tax(material, tonnage, country)
            if 'error' not in result:
                results.append({
                    'Country': country,
                    'Emissions (tCO₂/t)': result['predicted_emissions'],
                    'Total Tax (€)': result['predicted_tax_eur'],
                    'Per Tonne (€/t)': result['predicted_tax_eur'] / tonnage
                })
        
        if results:
            df_results = pd.DataFrame(results)
            
            # Find cheapest
            cheapest_idx = df_results['Total Tax (€)'].idxmin()
            cheapest_country = df_results.loc[cheapest_idx, 'Country']
            cheapest_tax = df_results.loc[cheapest_idx, 'Total Tax (€)']
            
            # Sort by price
            df_results = df_results.sort_values('Total Tax (€)')
            
            st.subheader("🏆 Best Country to Buy From:")
            st.success(f"""
            **{cheapest_country}** is the cheapest!
            
            Tax Cost: €{cheapest_tax:,.0f}
            """)
            
            st.subheader("📊 All Countries Comparison:")
            st.dataframe(df_results, use_container_width=True)
            
            st.subheader("💰 Cost Comparison Chart:")
            chart_data = df_results[['Country', 'Total Tax (€)']].set_index('Country')
            st.bar_chart(chart_data)
            
            # Show savings
            st.subheader("💵 Savings Opportunity:")
            most_expensive_idx = df_results['Total Tax (€)'].idxmax()
            most_expensive_country = df_results.loc[most_expensive_idx, 'Country']
            most_expensive_tax = df_results.loc[most_expensive_idx, 'Total Tax (€)']
            savings = most_expensive_tax - cheapest_tax
            
            st.info(f"""
            **By choosing {cheapest_country} instead of {most_expensive_country}:**
            
            You save: **€{savings:,.0f}**
            
            Percentage: **{(savings/most_expensive_tax)*100:.1f}% cheaper**
            """)

# PAGE 2: VIEW PAST SHIPMENTS
elif page == "📋 View Past Shipments":
    st.header("📋 View Past Shipments")
    st.markdown("See all shipments you've recorded")
    
    shipments = get_all_shipments()
    
    if shipments:
        df = pd.DataFrame(shipments)
        
        st.write(f"**Total shipments recorded: {len(df)}**")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Shipments", len(df))
        col2.metric("Total Weight", f"{df['tonnage'].sum():.0f}t")
        col3.metric("Average Emissions", f"{df['direct_emissions'].mean():.2f} tCO₂/t")
        col4.metric("Problem Shipments", len(df[df['audit_flag'] != 'PASS']))
        
        # Show table
        st.subheader("All Shipments")
        display_df = df[['shipment_id', 'material_type', 'tonnage', 'country_of_origin', 'direct_emissions', 'audit_flag']].copy()
        display_df.columns = ['ID', 'Material', 'Tonnage (t)', 'Country', 'Emissions (tCO₂/t)', 'Status']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Chart
        st.subheader("Materials Breakdown")
        material_chart = df['material_type'].value_counts()
        st.bar_chart(material_chart)
    else:
        st.info("No shipments yet. Load test data to get started!")

# PAGE 3: UNDERSTAND YOUR COST
elif page == "🔍 Understand Your Cost":
    st.header("🔍 Why Does It Cost This Much?")
    st.markdown("Let me explain what makes your shipment expensive or cheap")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        material = st.selectbox(
            "Material?",
            ["Steel", "Cement", "Aluminum", "Fertilizer", "Chemicals"],
            key="explain_material"
        )
    
    with col2:
        tonnage = st.number_input(
            "Tonnage?",
            min_value=10.0, max_value=500.0, value=100.0, step=10.0,
            key="explain_tonnage"
        )
    
    with col3:
        country = st.selectbox(
            "Country?",
            ["China", "India", "Russia", "Turkey", "Ukraine"],
            key="explain_country"
        )
    
    if st.button("📊 Explain Cost", key="explain", use_container_width=True):
        try:
            explanation = explain_prediction(material, tonnage, country)
            
            st.success("✅ Here's why it costs what it costs:")
            
            # Calculate and show cost
            predicted_emissions = explanation['prediction']
            predicted_tax = predicted_emissions * tonnage * 85
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Emissions", f"{predicted_emissions:.2f} tCO₂/t")
            col2.metric("Total Tax", f"€{predicted_tax:,.0f}")
            col3.metric("Per Tonne", f"€{predicted_tax/tonnage:,.0f}/t")
            
            # Show contributions - FILTERED to only selected
            st.subheader("What makes it expensive or cheap?")
            
            contrib = explanation['contributions']
            
            if contrib:
                # Filter to show only relevant features
                relevant = {}
                
                for feature, value in contrib.items():
                    # Keep if it matches selected material, country, or tonnage
                    if (material.lower() in feature.lower() or 
                        country.lower() in feature.lower() or 
                        'tonnage' in feature.lower()):
                        relevant[feature] = value
                
                # If no exact matches, show all
                if not relevant:
                    relevant = contrib
                
                # Display
                for feature, value in relevant.items():
                    if value > 0:
                        st.write(f"❌ **{feature.replace('material_type_', '').replace('country_of_origin_', '')}**: +{abs(value):.3f} (makes it EXPENSIVE)")
                    else:
                        st.write(f"✅ **{feature.replace('material_type_', '').replace('country_of_origin_', '')}**: {value:.3f} (makes it cheaper)")
            
            st.markdown("""
            **💡 How to read this:**
            - ❌ Red items = make cost go UP
            - ✅ Green items = make cost go DOWN
            
            **Tips to reduce cost:**
            - Try different material
            - Try different country
            - Buy less tonnage
            """)
        
        except Exception as e:
            st.error("⚠️ Train the AI first (go to 'Train the AI' tab)")

# PAGE 4: MODEL COMPARISON
elif page == "📊 Model Comparison":
    st.header("📊 Model Comparison")
    st.markdown("Compare 3 different AI models to see which is best")
    
    if st.button("🔍 Compare Models", use_container_width=True):
        with st.spinner("Comparing 3 models..."):
            try:
                results = compare_models()
                
                st.success("✅ Comparison complete!")
                
                # Display results
                results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy (R²)'])
                st.dataframe(results_df, use_container_width=True)
                
                # Chart
                st.bar_chart(results_df.set_index('Model'))
                
                # Winner
                best_model = max(results, key=results.get)
                st.success(f"🏆 Best Model: **{best_model}** ({results[best_model]:.1%} accurate)")
                
                st.info("""
                **What this means:**
                - XGBoost: Fast, accurate, good for small data
                - Random Forest: Slower but stable
                - Linear Regression: Simple but less accurate
                """)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# PAGE 5: CROSS-VALIDATION
elif page == "✅ Cross-Validation":
    st.header("✅ Cross-Validation Results")
    st.markdown("Evaluate model stability across 5 different data splits")
    
    if st.button("📊 Run Cross-Validation", use_container_width=True):
        with st.spinner("Running 5-fold validation..."):
            try:
                cv_results = cross_validate_model()
                
                st.success("✅ Validation complete!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean R²", f"{cv_results['mean']:.3f}")
                col2.metric("Std Dev", f"{cv_results['std']:.3f}")
                col3.metric("Min", f"{cv_results['min']:.3f}")
                col4.metric("Max", f"{cv_results['max']:.3f}")
                
                # Fold scores
                st.subheader("Scores per Fold:")
                fold_df = pd.DataFrame({
                    'Fold': [f'Fold {i+1}' for i in range(5)],
                    'R² Score': cv_results['fold_scores']
                })
                st.dataframe(fold_df, use_container_width=True)
                
                st.bar_chart(fold_df.set_index('Fold'))
                
                if cv_results['std'] < 0.05:
                    st.success("✅ Model is stable (low std = consistent)")
                else:
                    st.warning("⚠️ Model varies across folds")
                
                st.info("📊 **What this means:** Cross-validation tests model on 5 different data splits to ensure it generalizes well")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# PAGE 6: FEATURE IMPORTANCE
elif page == "⭐ Feature Importance":
    st.header("⭐ Feature Importance")
    st.markdown("Which factors matter most for carbon tax prediction?")
    
    if st.button("📊 Get Feature Importance", use_container_width=True):
        with st.spinner("Analyzing features..."):
            try:
                importance_df = get_feature_importance()
                
                st.success("✅ Analysis complete!")
                
                # Clean feature names
                importance_df['Feature'] = importance_df['Feature'].str.replace('material_type_', '').str.replace('country_of_origin_', '')
                
                # Table
                st.dataframe(importance_df, use_container_width=True)
                
                # Chart
                st.bar_chart(importance_df.set_index('Feature'))
                
                # Top 3
                top3 = importance_df.head(3)
                st.success(f"""
                **Top 3 Most Important Factors:**
                1. {top3.iloc[0]['Feature']}: {top3.iloc[0]['Importance']:.1%}
                2. {top3.iloc[1]['Feature']}: {top3.iloc[1]['Importance']:.1%}
                3. {top3.iloc[2]['Feature']}: {top3.iloc[2]['Importance']:.1%}
                """)
                
                st.info("💡 **What this means:** These are the factors that most influence carbon tax predictions")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# PAGE 7: ANOMALY DETECTION
elif page == "🚨 Anomaly Detection":
    st.header("🚨 Anomaly Detection")
    st.markdown("Find unusual shipments that need manual review")
    
    if st.button("🔍 Detect Anomalies", use_container_width=True):
        with st.spinner("Scanning for anomalies..."):
            try:
                anomalies = detect_anomalies()
                
                if anomalies:
                    st.warning(f"⚠️ Found {len(anomalies)} anomalies!")
                    
                    anomaly_df = pd.DataFrame(anomalies)
                    st.dataframe(anomaly_df, use_container_width=True)
                    
                    st.error("🔴 These shipments are unusual. Recommend manual audit.")
                    st.info("""
                    **Why flagged:**
                    - Emissions far from expected
                    - Unusual material-country combination
                    - Possible data entry error
                    """)
                else:
                    st.success("✅ No anomalies detected. All shipments look normal!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# PAGE 8: LEARNING CURVES
elif page == "📈 Learning Curves":
    st.header("📈 Learning Curves")
    st.markdown("Does model improve with more data?")
    
    if st.button("📊 Generate Learning Curves", use_container_width=True):
        with st.spinner("Generating curves..."):
            try:
                lc_data = plot_learning_curve_data()
                
                st.success("✅ Curves generated!")
                
                # Create dataframe for plotting
                lc_df = pd.DataFrame({
                    'Data Size': [int(x) for x in lc_data['train_sizes']],
                    'Training Accuracy': lc_data['train_mean'],
                    'Validation Accuracy': lc_data['val_mean']
                })
                
                st.line_chart(lc_df.set_index('Data Size'))
                
                # Interpretation
                train_final = lc_data['train_mean'][-1]
                val_final = lc_data['val_mean'][-1]
                gap = train_final - val_final
                
                if gap > 0.1:
                    st.warning(f"⚠️ Overfitting detected (gap: {gap:.3f})")
                    st.info("Model is too specialized on training data. Needs more diverse data.")
                else:
                    st.success(f"✅ Model generalizes well (gap: {gap:.3f})")
                    st.info("Model performs similarly on training and validation data.")
                
                st.markdown("""
                **How to read:**
                - Blue line = accuracy on training data
                - Orange line = accuracy on new data
                - Both going UP = model improving with more data
                - Large gap = overfitting
                - Small gap = good generalization
                """)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# PAGE 9: TRAIN THE AI
elif page == "🧠 Train the AI":
    st.header("🧠 Train the AI")
    st.markdown("The AI learns from past shipments to make better predictions")
    
    shipments = get_all_shipments()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📊 Records available: {len(shipments)}")
        
        if st.button("🔄 Train AI Now", use_container_width=True):
            with st.spinner("Training AI (30 seconds)..."):
                try:
                    r2 = train_cbam_model()
                    st.success(f"""
                    ✅ Training Complete!
                    
                    **Accuracy: {r2:.1%}**
                    
                    The AI is now {r2:.1%} accurate at predicting carbon tax costs.
                    """)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.info(f"""
        **How it works:**
        - AI learns from {len(shipments)} past shipments
        - Recognizes patterns (Steel from China is expensive, etc.)
        - Makes predictions for new shipments
        
        **More data = Better predictions**
        """)

# PAGE 10: LOAD TEST DATA
elif page == "📥 Load Test Data":
    st.header("📥 Load Test Data")
    st.markdown("Load fake shipments to practice with")
    
    if st.button("➕ Add 15 Test Shipments", use_container_width=True):
        with st.spinner("Adding test data..."):
            mock = generate_mock_shipments(15)
            count = 0
            for record in mock:
                if insert_shipment(record):
                    count += 1
            st.success(f"✅ Added {count} test shipments!")
            st.info(f"Now you have {len(get_all_shipments())} total shipments. Go to 'Train the AI' to train.")