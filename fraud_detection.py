import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_detection_random_forest.pkl")

st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and use the prediction button")
st.divider()

# Input fields
transaction_type = st.selectbox(
    "Transaction Type", 
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceorg = st.number_input("Old Balance (Sender)", min_value=0.0, value=0.0)
newbalanceorig = st.number_input("New Balance (Sender)", min_value=0.0, value=0.0)
oldbalancedest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalancedest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# Predict button
if st.button("Predict"):
    # Create input DataFrame with feature names matching training data
    type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
    type_DEBIT = 1 if transaction_type == "DEBIT" else 0
    type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
    type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0
    input_data = pd.DataFrame([{
        'step': 1,
        'amount': amount,
        'oldbalanceOrg': oldbalanceorg,
        'newbalanceOrig': newbalanceorig,
        'oldbalanceDest': oldbalancedest,
        'newbalanceDest': newbalancedest,
        'type_CASH_OUT': type_CASH_OUT,
        'type_DEBIT': type_DEBIT,
        'type_PAYMENT': type_PAYMENT,
        'type_TRANSFER': type_TRANSFER
    }])
    # Model prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    if prediction == 1:
        st.error("⚠️ Warning: This transaction is predicted as **FRAUD**!")
    else:
        st.success("✅ This transaction appears to be **LEGITIMATE (Not Fraud)**.")