import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

model = joblib.load("fraud_model.pkl")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🤖 Prediction", "ℹ About"])

if page == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection")
    st.write("This application predicts whether a credit card transaction is Fraud or Normal using a Machine Learning model.")
    st.success("Upload a CSV file in the Prediction page to detect fraud.")

elif page == "🤖 Prediction":
    st.title("🤖 Fraud Prediction")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        if st.button("Predict"):
            pred = model.predict(df)
            prob = model.predict_proba(df)

            result = df.copy()
            result["Prediction"] = ["Fraud" if i == 1 else "Normal" for i in pred]
            result["Fraud Probability"] = prob[:,1]

            st.success("Prediction Completed")
            st.dataframe(result)

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Result",
                csv,
                "prediction_result.csv",
                "text/csv"
            )

elif page == "ℹ About":
    st.title("ℹ About")
    st.write("""
    **Project:** Credit Card Fraud Detection

    **Algorithm:** Random Forest Classifier

    **Tools Used:**
    - Python
    - Streamlit
    - Pandas
    - Scikit-Learn
    - Joblib

    Developed by **Simin Maner**
    """)