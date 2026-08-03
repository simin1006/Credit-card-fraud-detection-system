import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

model = joblib.load("fraud_model.pkl")

st.sidebar.title("💳 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "🤖 Fraud Prediction", "ℹ About"]
)

# ---------------- HOME ----------------

if page == "🏠 Home":

    st.title("💳 Credit Card Fraud Detection")

    st.write("Welcome to the Credit Card Fraud Detection System.")

    st.info("Go to the Fraud Prediction page to check transactions.")

# ---------------- PREDICTION ----------------

elif page == "🤖 Fraud Prediction":

    st.title("🤖 Fraud Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.dataframe(df.head())

        if st.button("Predict Transactions"):

            if "Class" in df.columns:
                df = df.drop("Class", axis=1)

            prediction = model.predict(df)
            probability = model.predict_proba(df)

            result = df.copy()

            result["Transaction Status"] = [
                "❌ Failed (Fraud)"
                if i == 1
                else "✅ Successful"
                for i in prediction
            ]

            result["Fraud Probability (%)"] = (
                probability[:,1] * 100
            ).round(2)

            st.success("Prediction Completed Successfully ✅")

            total = len(result)
            fraud = (result["Transaction Status"]=="❌ Failed (Fraud)").sum()
            success = total - fraud

            c1,c2,c3 = st.columns(3)

            with c1:
                st.metric("Total", total)

            with c2:
                st.metric("Successful", success)

            with c3:
                st.metric("Failed", fraud)

            if fraud > 0:
                st.error(f"{fraud} Fraud Transaction(s) Detected.")
            else:
                st.success("All Transactions are Safe.")

            st.subheader("Transaction Report")
            st.dataframe(result)

            csv = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Report",
                data=csv,
                file_name="transaction_report.csv",
                mime="text/csv"
            )

# ---------------- ABOUT ----------------

elif page == "ℹ About":

    st.title("ℹ About")

    st.write("""
This project predicts fraudulent credit card transactions
using a Machine Learning model.
""")
