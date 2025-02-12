# -*- coding: utf-8 -*-
"""
 STREAMLIT INTERFACE
"""

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the trained XGBoost model
model = joblib.load(r"C:\Users\priya\OneDrive\Desktop\PRIYA\internship at 360 digitmg\CREDIT CARD\xgboost_fraud_model.pkl")

# Define feature columns (must match training features)
feature_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                   'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                   'Amount']

# Streamlit UI
st.title("💳 Credit Card Fraud Detection 🚀")
st.divider()
st.subheader("Upload transaction data to check for fraud.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check if all required columns are present
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
       # Convert DataFrame to DMatrix (Fix XGBoost Error)
        dtest = xgb.DMatrix(df[feature_columns])  
        
        # Make predictions
        predictions = model.predict(dtest)
        df["Fraud Probability"] = predictions
        df["Fraud Prediction"] = (predictions > 0.5).astype(int)  # Convert probabilities to 0/1

        # Show results
        st.write("### Prediction Results:")
        st.dataframe(df)
        
        # 📊 **Graph 1: Fraud vs. Non-Fraud Counts**
        st.write("### Fraud vs. Non-Fraud Transactions")
        fraud_counts = df["Fraud Prediction"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette=["blue", "red"], ax=ax)
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        ax.set_ylabel("Number of Transactions")
        ax.set_xlabel("Transaction Type")
        st.pyplot(fig)

        # 📊 **Graph 2: Fraud Probability Distribution**
        st.write("### Fraud Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Fraud Probability"], bins=30, kde=True, color="purple", ax=ax)
        ax.set_xlabel("Fraud Probability")
        ax.set_ylabel("Number of Transactions")
        st.pyplot(fig)

        # 📊 **Graph 3: Feature Importance from XGBoost**
        st.write("### Feature Importance (XGBoost)")
        fig, ax = plt.subplots(figsize=(8,5))
        xgb.plot_importance(model, height=0.6, title="Feature Importance", ax=ax, color="green")
        st.pyplot(fig)

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", data=csv, file_name="fraud_predictions.csv", mime="text/csv")


