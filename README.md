# 🛡️ E-commerce Fraud Detection System

> Machine Learning-based fraud detection for Wildberries & Flip.kz marketplaces

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## 📊 Overview

Fraud detection system analyzing **5,862 products** from **2,578 sellers** across Wildberries and Flip.kz marketplaces. Combines rule-based indicators with ML classification and K-Means clustering.

### Key Features

- 🎯 **Multi-indicator Detection**: 5 fraud signals (fake reviews, fraud seller, low quality, price manipulation)
- 🤖 **ML Models**: Logistic Regression, Random Forest, Gradient Boosting
- 👥 **Seller Segmentation**: K-Means clustering into 3 risk profiles
- 📈 **Fraud Score**: 0-100 composite risk score

## 🔍 Fraud Detection Logic

```python
is_fraud_seller = (
    (seller_age_months < 6 AND seller_total_sold < 10) OR  # New + inactive
    (seller_total_sold > 0 AND feedbacks == 0) OR          # No feedback
    (|price_zscore| > 3)                                    # Price anomaly
)
```

**Why no `seller_rating`?** To avoid data leakage - rating shouldn't predict itself!

## 🤖 Model Results

| Model               | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------- | -------- | ------- |
| **Random Forest**   | 99.91%   | 100.00%   | 100.00% | 100.00%  | 100.00% |
| Gradient Boosting   | 99.91%   | 100.00%   | 98.65%  | 99.32%   | 100.00% |
| Logistic Regression | 89.17%   | 36.46%    | 94.59%  | 52.63%   | 96.69%  |

**Best Model**: Random Forest (perfect classification)

## 👥 Seller Clusters

| Cluster | Size          | Fraud Rate | Risk Level   | Characteristics             |
| ------- | ------------- | ---------- | ------------ | --------------------------- |
| **0**   | 171 (2.9%)    | 22.8%      | 🌟 Premium   | High-price, trusted sellers |
| **1**   | 4,901 (83.6%) | 7.4%       | ✅ Reliable  | Standard mainstream sellers |
| **2**   | 790 (13.5%)   | 28.0%      | ⚠️ High Risk | Old accounts, low rating    |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook ecommerce_fraud_detection_final.ipynb
```

### Basic Usage

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('all_products.csv')

# Feature engineering
df['sales_per_month'] = df['seller_total_sold'] / (df['seller_age_months'] + 1)
df['feedback_ratio'] = df['feedbacks'] / (df['seller_total_sold'] + 1)

# Train model
features = ['price_rub', 'seller_total_sold', 'seller_age_months',
            'feedbacks', 'price_zscore', 'sales_per_month', 'feedback_ratio']
X = df[features].values
y = df['is_fraud_seller'].values

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X, y)

# Predict fraud probability
fraud_prob = model.predict_proba(X)[:, 1]
```

## 📈 Key Insights

- **Account age + sales activity** are strongest fraud predictors
- **Young sellers (<6 months)** with <10 sales are high risk
- **Missing feedback** despite sales indicates fraud
- **Fraud rate**: 6.35% (372/5,862 products)

## 🎯 Recommendations

- Flag: age <6mo + sales <10 → manual review
- Investigate: sales >0 BUT feedbacks =0 → fraud
- Monitor: |price_zscore| >3 → price manipulation
- Deploy: Random Forest for real-time scoring

## 🛠️ Tech Stack

**Python 3.8+** | **Pandas** | **NumPy** | **Scikit-learn** | **Matplotlib** | **Seaborn**

---

**Built with ❤️ for safer e-commerce**
