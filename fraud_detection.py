import streamlit as st
import pandas as pd
import joblib
import sklearn

# --------------------------------------------------------------------
# 🧠 Load model safely
# --------------------------------------------------------------------
st.title("💸 Digital Payment Fraud Detection App")
st.markdown("Enter transaction details and click **Predict** to check if it's fraudulent.")
st.divider()

st.info(f"✅ Using scikit-learn version: {sklearn.__version__}")

model_path = "fraud_detection_pipeline.pkl"

try:
    model = joblib.load(model_path)
    st.success(f"Model loaded successfully from `{model_path}`")
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# --------------------------------------------------------------------
# 📋 User input
# --------------------------------------------------------------------
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# Ensure order matches model training columns
input_data = pd.DataFrame([{
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "type": transaction_type
}])[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type"]]

st.write("### Input Data Preview")
st.dataframe(input_data)

# --------------------------------------------------------------------
# 🚀 Prediction
# --------------------------------------------------------------------
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.error("🚨 **Fraudulent Transaction Detected!**")
        else:
            st.success("✅ **Legitimate Transaction.**")

        if probability is not None:
            st.caption(f"Model Confidence: {probability:.2%}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

st.divider()
st.caption("Developed with ❤️ using Streamlit & scikit-learn 1.4.2")
