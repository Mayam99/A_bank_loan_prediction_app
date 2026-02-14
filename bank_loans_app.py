import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD MODEL COMPONENTS ---
@st.cache_resource # Keeps model in memory for faster performance
def load_assets():
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

model, scaler, features = load_assets()

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Banking AI | Loan Predictor", layout="centered")

st.title("🏦 Banking AI: Loan Repayment Predictor")
st.markdown("This system uses Machine Learning to assess the probability of loan repayment.")
st.divider()

# --- 3. INPUT FIELDS ---
st.subheader("📝 Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Customer Age", 18, 100, 30)
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

with col2:
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.0)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
    # Suggestion: Add Income here if it was in your original dataset
    # income = st.number_input("Annual Income ($)", 0, 1000000, 50000)

# --- 4. PREDICTION LOGIC ---
st.markdown("### Prediction Engine")
if st.button("Run Risk Analysis", use_container_width=True):
    
    # a. Create Input DataFrame
    input_data = {
        'age': age,
        'credit_score': credit_score,
        'interest_rate': interest_rate / 100, # Convert to decimal
        'loan_term': loan_term,
        'marital_status': marital_status
    }
    
    input_df = pd.DataFrame([input_data])
    
    # b. Encoding & Feature Alignment
    input_encoded = pd.get_dummies(input_df).reindex(columns=features, fill_value=0)
    
    # c. Scaling
    input_scaled = scaler.transform(input_encoded)
    
    # d. Prediction & Probability
    prob_paid = model.predict_proba(input_scaled)[0][1]
    
    # --- 5. DECISION LOGIC (Bank Policy) ---
    
    # We set a strict policy: Credit Score must be > 580 AND Probability > 70%
    min_credit_allowed = 580
    approval_threshold = 0.70 

    st.divider()
    
    if credit_score < min_credit_allowed:
        st.error(f"❌ **REJECTED: UNACCEPTABLE CREDIT SCORE**")
        st.info(f"The bank requires a minimum credit score of {min_credit_allowed}. Current: {credit_score}")
    
    elif prob_paid < approval_threshold:
        st.warning(f"❌ **REJECTED: HIGH RISK PROFILE**")
        st.write(f"The model confidence for repayment is only **{prob_paid:.1%}**, which is below our **{approval_threshold:.0%}** safety threshold.")
    
    else:
        st.success(f"✅ **LOAN APPROVED**")
        st.balloons()
        st.write(f"Repayment Confidence: **{prob_paid:.1%}**")

# --- 6. FOOTER DEBUGGER ---
with st.expander("ℹ️ System Diagnostics"):
    st.write("Target Features:", features)
    st.write("Raw Probabilities (Default vs Paid):", model.predict_proba(input_scaled))