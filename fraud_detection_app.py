import streamlit as st
import pandas as pd
import joblib

st.title("💰 Fraud Detection App")

# Load trained model and encoders
model = joblib.load("fraud_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.subheader("🔎 Predict Fraud for a New Transaction")

# Input fields
step = st.number_input("Step", min_value=1, value=1)
type_input = st.selectbox("Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT'])
amount = st.number_input("Amount", min_value=0.0, value=100.0)
oldbalanceOrg = st.number_input("Old Balance Orig", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance Orig", min_value=0.0, value=4900.0)
oldbalanceDest = st.number_input("Old Balance Dest", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance Dest", min_value=0.0, value=0.0)

# Encode 'type' using saved encoder
type_encoded = label_encoders['type'].transform([type_input])[0]

# Prepare input DataFrame
input_data = pd.DataFrame([[step, type_encoded, amount, 0, oldbalanceOrg,
                            newbalanceOrig, 0, oldbalanceDest, newbalanceDest]],
                          columns=['step', 'type', 'amount', 'nameOrig',
                                   'oldbalanceOrg', 'newbalanceOrig', 'nameDest',
                                   'oldbalanceDest', 'newbalanceDest'])

# Predict button
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
