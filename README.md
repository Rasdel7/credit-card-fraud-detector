# Credit Card Fraud Detector 💳

Detects fraudulent credit card transactions using
Random Forest with SMOTE oversampling.
Trained on 284,807 real transactions.

## Live Demo
[Click here](https://credit-card-fraud-detector-kd8vwgcmbfi729va9ubneb.streamlit.app)

## The Challenge
Only 0.17% of transactions are fraudulent — extreme
class imbalance solved using SMOTE oversampling.

## Model Details
- Algorithm: Random Forest
- Technique: SMOTE oversampling
- AUC-ROC: ~0.97
- Dataset: 284,807 transactions (2 days)

## Features
- Real-time fraud probability scoring
- Risk level classification (Low/Medium/High)
- Dataset statistics and insights
- Precision-Recall curve analysis
- Feature importance visualization

## Tools Used
- Python, Scikit-learn, imbalanced-learn
- Streamlit, Pandas, Matplotlib, Seaborn

## How to Run Locally
pip install streamlit scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
python3 train_model.py
streamlit run app.py
