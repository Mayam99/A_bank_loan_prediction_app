import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved components
# Ensure these files are in the same GitHub folder as app.py
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('features.pkl')

st.set_page_config(page_title="Loan AI Predictor", layout="wide")

st.title("🏦 Banking AI: Loan Repayment Predictor")
st.markdown("---")

# 2. User Input UI
st.sidebar.header("Model Settings")
# This threshold helps handle imbalance. 
# If the model is too "optimistic", move this to 0.7 or 0.8
threshold = st.sidebar.slider("Approval Threshold", 0.0, 1.0, 0.5, 0.1)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Info")
    age = st.number_input("Age", 18, 100, 30)
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    
with col2:
    st.subheader("Loan Details")
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 35.0, 10.0)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])

# 3. Prediction Logic
if st.button("Analyze Loan Risk", use_container_width=True):
    # Prepare the input dictionary
    # IMPORTANT: Ensure keys match your original CSV column names exactly
    input_data = {
        'age': age,
        'credit_score': credit_score,
        'interest_rate': interest_rate / 100,
        'loan_term': loan_term,
        'marital_status': marital_status
    }
    
    # Transform input to match training format
    input_df = pd.DataFrame([input_data])
    
    # One-Hot Encoding and Column Alignment
    input_encoded = pd.get_dummies(input_df).reindex(columns=model_features, fill_value=0)
    
    # Scaling
    input_scaled = scaler.transform(input_encoded)
    
    # Get Probability
    # probability[0] is chance of 'Default', probability[1] is chance of 'Paid'
    prob_paid = model.predict_proba(input_scaled)[0][1]

    st.markdown("### Result")
    # Apply custom threshold for imbalanced data
    if prob_paid >= threshold:
        st.success(f"✅ **APPROVED** - Confidence: {prob_paid:.2%}")
    else:
        st.error(f"❌ **REJECTED** - High Risk (Repayment Confidence: {prob_paid:.2%})")

    # 4. Debugging Section (Hidden by default)
    with st.expander("See technical details (Debug)"):
        st.write("Processed Input (Scaled):", input_scaled)
        st.write("Feature Order used by Model:", model_features)