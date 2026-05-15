import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fraud Detector",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detector")
st.markdown("Detect fraudulent transactions "
            "using ML trained on 284,807 real transactions.")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl',  'rb') as f:
        model  = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    st.success(
        "✅ Model loaded — "
        "trained on 284,807 transactions")
except:
    st.error("Run train_model.py first!")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Detect Fraud",
    "📊 Dataset Insights",
    "📈 Model Performance"
])

# Tab 1 — Detect
with tab1:
    st.markdown("### Enter Transaction Details")
    st.info(
        "💡 In real systems V1-V28 are PCA "
        "components of anonymized transaction "
        "features. Adjust them to test the model.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Transaction Info")
        amount = st.number_input(
            "Transaction Amount ($):",
            min_value=0.0,
            max_value=30000.0,
            value=150.0, step=0.01)
        time   = st.number_input(
            "Time (seconds from first transaction):",
            min_value=0.0,
            max_value=200000.0,
            value=50000.0)

        st.markdown("#### 🔢 Key Features (V1-V14)")
        v_vals_1 = {}
        for i in range(1, 15):
            v_vals_1[f'V{i}'] = st.slider(
                f"V{i}:",
                -10.0, 10.0, 0.0, 0.1,
                key=f"v{i}"
            )

    with col2:
        st.markdown("#### 🔢 Features (V15-V28)")
        v_vals_2 = {}
        for i in range(15, 29):
            v_vals_2[f'V{i}'] = st.slider(
                f"V{i}:",
                -10.0, 10.0, 0.0, 0.1,
                key=f"v{i}"
            )

        st.markdown("#### 🎯 Quick Test Scenarios")
        if st.button("Load Normal Transaction"):
            st.info(
                "Set all V values to 0, "
                "Amount to ~50-200. "
                "Typical normal transaction.")
        if st.button("Load Suspicious Transaction"):
            st.warning(
                "High negative V1, V2, V3 "
                "and high amount typically "
                "indicate fraud patterns.")

    if st.button("🔍 Detect Fraud",
                 type="primary"):
        # Build input
        input_dict = {'Time': time,
                      'Amount': amount}
        input_dict.update(v_vals_1)
        input_dict.update(v_vals_2)

        input_df = pd.DataFrame([input_dict])

        # Scale Amount and Time
        input_df['Amount'] = scaler.transform(
            input_df[['Amount']])
        input_df['Time']   = scaler.transform(
            input_df[['Time']])

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(
            input_df)[0]

        fraud_prob  = probability[1] * 100
        normal_prob = probability[0] * 100

        st.markdown("---")

        if prediction == 1:
            st.markdown(
                "<h2 style='color:#e74c3c; "
                "text-align:center'>"
                "🚨 FRAUDULENT TRANSACTION "
                "DETECTED</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color:#2ecc71; "
                "text-align:center'>"
                "✅ NORMAL TRANSACTION</h2>",
                unsafe_allow_html=True
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Probability",
                  f"{fraud_prob:.2f}%")
        c2.metric("Normal Probability",
                  f"{normal_prob:.2f}%")
        c3.metric("Risk Level",
                  "HIGH" if fraud_prob > 50
                  else "LOW" if fraud_prob < 20
                  else "MEDIUM")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(8, 2))
        color = '#e74c3c' if fraud_prob > 50 \
                else '#f39c12' if fraud_prob > 20 \
                else '#2ecc71'
        ax.barh(['Risk'], [fraud_prob],
                color=color, height=0.4)
        ax.barh(['Risk'],
                [100 - fraud_prob],
                left=[fraud_prob],
                color='#ecf0f1', height=0.4)
        ax.set_xlim(0, 100)
        ax.set_title(
            f'Fraud Risk Score: '
            f'{fraud_prob:.1f}%',
            fontsize=12)
        ax.set_xlabel('Risk (%)')
        plt.tight_layout()
        st.pyplot(fig)

# Tab 2 — Dataset Insights
with tab2:
    st.markdown("### 📊 Dataset Statistics")

    stats = pd.DataFrame({
        'Statistic': [
            'Total transactions',
            'Normal transactions',
            'Fraudulent transactions',
            'Fraud rate',
            'Time period',
            'Features',
            'Amount range'
        ],
        'Value': [
            '284,807',
            '284,315 (99.83%)',
            '492 (0.17%)',
            '0.1727%',
            '2 days',
            '30 (Time, Amount, V1-V28)',
            '$0 — $25,691'
        ]
    })
    st.dataframe(stats,
                 use_container_width=True,
                 hide_index=True)

    st.markdown("### 🎯 Why This Is Hard")
    st.markdown("""
    This dataset has **extreme class imbalance**:
    - For every 1 fraud case there are 577 normal ones
    - Standard accuracy would be 99.83% by
      just predicting everything as normal
    - We use **SMOTE** to oversample the minority
      class and **AUC-ROC** as our metric
    - **Precision-Recall** curve is more
      informative than ROC for imbalanced data
    """)

    # Show charts if they exist
    charts = ['class_distribution.png']
    for chart in charts:
        if os.path.exists(chart):
            st.image(chart,
                     use_column_width=True)

# Tab 3 — Model Performance
with tab3:
    st.markdown("### 📈 Model Performance")

    charts = [
        'confusion_matrix.png',
        'precision_recall.png',
        'feature_importance.png'
    ]
    for chart in charts:
        if os.path.exists(chart):
            st.image(chart,
                     use_column_width=True)
        else:
            st.info(
                f"Run train_model.py to "
                f"generate {chart}")

    perf = pd.DataFrame({
        'Metric': [
            'Algorithm',
            'Technique',
            'AUC-ROC',
            'Training samples (after SMOTE)',
            'Test samples',
            'Fraud cases in test'
        ],
        'Value': [
            'Random Forest',
            'SMOTE oversampling',
            '~0.97',
            '~454,902',
            '~56,962',
            '~98'
        ]
    })
    st.dataframe(perf,
                 use_container_width=True,
                 hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Credit Card Fraud Detector | "
    "284,807 real transactions analyzed"
)