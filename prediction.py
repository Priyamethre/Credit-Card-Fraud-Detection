# -*- coding: utf-8 -*-
"""

"""

import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb

# Load the model
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

         # Save results to session state
        st.session_state["predicted_df"] = df  

    

# Show results
        st.write("### Prediction Results:")
        st.dataframe(df)
        
        
        
        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

