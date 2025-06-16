import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('credwise_model.pkl')

st.title("💳 CredWise - Loan Interest Rate Predictor")
st.subheader("Your Credit, Your Terms — Powered by AI")

# Input UI
with st.form("input_form"):
    income = st.number_input("Annual Income (₹ Lakhs)", min_value=1.0, step=0.5)
    loan = st.number_input("Loan Amount (₹ Lakhs)", min_value=1.0, step=0.5)
    tenure = st.slider("Loan Tenure (Years)", 1, 10, 5)
    score = st.slider("CIBIL Score", 300, 900, 750)
    dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 0.50, 25.0)
    tier = st.selectbox("City Tier", [1, 2, 3])

    submit = st.form_submit_button("Predict Interest Rate")

if submit:
    df = pd.DataFrame([{
        'Annual_Income': income,
        'Loan_Amount': loan,
        'Loan_Tenure': tenure,
        'CIBIL_Score': score,
        'DTI_Ratio': dti/100,
        'City_Tier': tier
    }])

    result = model.predict(df)[0]
    out=abs(result*100)
    st.success(f"Predicted Interest Rate: {out:.2f}%")
