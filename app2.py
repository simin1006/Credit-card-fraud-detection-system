import streamlit as st
import random

st.set_page_config(
    page_title="Credit Card Transaction Checker",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Transaction Checker")
st.write("Enter the transaction details below.")

transaction_id = st.text_input("Transaction ID")
amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=100.0)
hour = st.slider("Transaction Time (Hour)", 0, 23, 12)

if st.button("Check Transaction"):

    if amount > 50000 or hour >= 23 or hour <= 4:
        status = "❌ Failed"
        risk = "🔴 High Risk"
        probability = random.randint(80, 99)

    elif amount > 10000:
        status = "⚠️ Under Review"
        risk = "🟡 Medium Risk"
        probability = random.randint(40, 79)

    else:
        status = "✅ Successful"
        risk = "🟢 Low Risk"
        probability = random.randint(1, 39)

    st.success("Transaction Analysis Completed")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Status", status)

    with col2:
        st.metric("Fraud Probability", f"{probability}%")

    st.info(f"Risk Level: {risk}")

    if status == "✅ Successful":
        st.success("Transaction Completed Successfully.")

    elif status == "⚠️ Under Review":
        st.warning("Transaction is Under Review.")

    else:
        st.error("Transaction Failed due to Suspected Fraud.")

st.markdown("---")