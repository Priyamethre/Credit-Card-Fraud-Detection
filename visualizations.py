# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:48:14 2025

@author: priya
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

# Load the trained XGBoost model
model = joblib.load(r"C:\Users\priya\OneDrive\Desktop\PRIYA\internship at 360 digitmg\CREDIT CARD\xgboost_fraud_model.pkl")

st.title("📊 Fraud Detection Visualizations 🚀")
st.divider()

# Load predictions from session state
if "predicted_df" not in st.session_state:
    st.warning("⚠️ No predictions made yet! Please upload a CSV on the Prediction Page.")
else:
    df = st.session_state["predicted_df"]

    # Convert 'Fraud Prediction' to categorical labels
    df["Fraud Prediction"] = df["Fraud Prediction"].map({0: "Non-Fraud", 1: "Fraud"})

    # 1️⃣ **Fraud vs. Non-Fraud Counts**
    st.subheader("🔵 Fraud vs. Non-Fraud Transactions")
    fig, ax = plt.subplots()
    sns.countplot(x="Fraud Prediction", data=df, palette=["blue", "red"], ax=ax)
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 2️⃣ **Fraud Probability Distribution**
    st.subheader("📈 Fraud Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Fraud Probability"], bins=20, kde=True, color="purple", ax=ax)
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Number of Transactions")
    st.pyplot(fig)

    # 3️⃣ **Feature Importance from XGBoost**
    st.subheader("🚀 Feature Importance from XGBoost")
    importance = model.get_score(importance_type="weight")
    importance_df = pd.DataFrame({"Feature": list(importance.keys()), "Importance": list(importance.values())})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_r", ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature Name")
    st.pyplot(fig)

    # 4️⃣ **Show the DataFrame**
    st.write("### 📜 Prediction Data Overview")
    st.dataframe(df)
