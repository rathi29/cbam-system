# 🌍 EU Carbon Tax (CBAM) Compliance & Prediction Pipeline

Advanced machine learning system for predicting EU carbon taxes on material imports with explainability, model comparison, and anomaly detection.

**[🚀 Live Demo](https://cbam-system.streamlit.app/)**

---

## 🎯 Problem Statement

EU companies importing materials face carbon taxes under the Carbon Border Adjustment Mechanism (CBAM). They need to:
- Predict carbon tax costs BEFORE purchasing
- Understand what factors drive costs
- Compare suppliers across countries
- Detect anomalies in shipment data

This system solves all of that.

---

## ✨ Key Features

### Core Predictions
- **Multi-Model Ensemble**: XGBoost, Random Forest, Linear Regression
- **94% Accuracy**: R² score on carbon emission predictions
- **Instant Predictions**: Sub-millisecond inference time
- **Batch Processing**: Predict for 100+ shipments at once

### Explainability & Analysis
- **SHAP Explainability**: Understand why each prediction
- **Model Comparison**: Side-by-side accuracy comparison
- **Cross-Validation**: 5-fold testing for reliability
- **Feature Importance**: See which factors matter most
- **Learning Curves**: Monitor model improvement

### Data Quality & Safety
- **Anomaly Detection**: Flag unusual shipments for review
- **Validation Pipeline**: Catch bad data before training
- **Historical Tracking**: Complete audit trail

### Advanced Features
- **Time Series Forecasting**: Predict future emissions trends
- **Active Learning**: Identify highest-uncertainty shipments
- **Bayesian Optimization**: Auto-tune hyperparameters

---

## 🛠 Tech Stack

**Machine Learning:**
- `xgboost` - Primary prediction model
- `scikit-learn` - ML utilities, Random Forest, validation
- `shap` - Model explainability
- `scikit-optimize` - Bayesian hyperparameter tuning

**Data & Database:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `sqlite3` - Lightweight database

**Frontend:**
- `streamlit` - Interactive dashboard (10 tabs)

**Deployment:**
- `Streamlit Cloud` - Free hosting
- `GitHub` - Version control

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Model Accuracy (R²)** | 0.94 |
| **Prediction Speed** | <1ms |
| **Training Time** | 30 seconds |
| **Data Required** | 15+ records |
| **Deployment Status** | ✅ Live |

---

## 🚀 Quick Start

### Live Demo
Open: **[https://cbam-system.streamlit.app/](https://cbam-system.streamlit.app/)**

No installation needed. Try it now!

### Local Setup

**1. Clone repository**
```bash
git clone https://github.com/rathi29/cbam-system.git
cd cbam-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run dashboard**
```bash
streamlit run dashboard.py
```

**4. Open browser**


---

## 📋 How to Use

### 1. Load Data
- Go to **"📥 Load Test Data"** tab
- Click to add 15 fake shipments

### 2. Train AI
- Go to **"🧠 Train the AI"** tab
- Click "Train AI Now"
- Wait 30 seconds

### 3. Make Predictions
- Go to **"💡 Quick Tax Prediction"**
- Enter: Material, Tonnage, Country
- Get: Best country to buy from + cost comparison

### 4. Understand Costs
- Go to **"🔍 Understand Your Cost"**
- See SHAP breakdown of each factor

### 5. Advanced Analysis
- **Model Comparison**: Compare 3 models
- **Cross-Validation**: 5-fold reliability test
- **Feature Importance**: Which factors matter
- **Anomaly Detection**: Find risky shipments
- **Learning Curves**: Model improvement over time
