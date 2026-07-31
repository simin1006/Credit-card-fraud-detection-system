import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import gdown

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main{
    background:#F5F7FB;
}

h1,h2,h3{
    color:#0F172A;
}

[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stSidebar"] *{
    color:white;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.10);
}

.footer{
text-align:center;
padding:15px;
font-size:15px;
color:gray;
}

</style>
""",unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

@st.cache_data
def load_data():

    if not os.path.exists("cleaned_creditcard.csv"):

        file_id = "1kY_qBgRNIfd9YjWj_CCIrOTJY2ObccvI"

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, "cleaned_creditcard.csv", quiet=False)

    df = pd.read_csv("cleaned_creditcard.csv")

    return df

df = load_data()

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

@st.cache_resource
def load_model(): 

    model=joblib.load("fraud_model.pkl")
    return model

model=load_model()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("💳 Credit Card Fraud Detection")

page=st.sidebar.radio(

"Navigation",

[
"🏠 Dashboard",
"📂 Dataset",
"📊 EDA",
"📈 Visualizations",
"🤖 Prediction",
"ℹ About"
]

)

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.info(f"""
Rows : {df.shape[0]}

Columns : {df.shape[1]}
""")

# ----------------------------------------------------
# DASHBOARD
# ----------------------------------------------------

if page=="🏠 Dashboard":

    st.title("💳 Credit Card Fraud Detection Dashboard")

    st.write("Professional Machine Learning Dashboard")

    total=df.shape[0]

    fraud=df["Class"].sum()

    normal=total-fraud

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Total Transactions",f"{total:,}")

    c2.metric("Features",df.shape[1])

    c3.metric("Fraud",fraud)

    c4.metric("Normal",normal)

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10),use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe(),use_container_width=True)

# ----------------------------------------------------
# DATASET PAGE
# ----------------------------------------------------

elif page=="📂 Dataset":

    st.title("Dataset Overview")

    st.write("Shape :",df.shape)

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

    st.subheader("Data Types")

    st.dataframe(df.dtypes.astype(str))
# ----------------------------------------------------
# EDA PAGE
# ----------------------------------------------------

elif page=="📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    st.markdown("### Dataset Preview")

    rows = st.slider(
        "Select Number of Rows",
        5,
        50,
        10
    )

    st.dataframe(df.head(rows), use_container_width=True)

    st.markdown("---")

    st.markdown("### Class Distribution")

    class_count = df["Class"].value_counts().reset_index()
    class_count.columns = ["Transaction", "Count"]

    class_count["Transaction"] = class_count["Transaction"].replace({
        0: "Normal",
        1: "Fraud"
    })

    fig = px.pie(
        class_count,
        names="Transaction",
        values="Count",
        hole=0.45,
        title="Fraud vs Normal Transactions"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Transaction Amount Distribution")

    fig = px.histogram(
        df,
        x="Amount",
        nbins=60,
        title="Amount Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Transaction Time Distribution")

    fig = px.histogram(
        df,
        x="Time",
        nbins=60,
        title="Time Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Amount Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Minimum", round(df["Amount"].min(),2))
    c2.metric("Maximum", round(df["Amount"].max(),2))
    c3.metric("Average", round(df["Amount"].mean(),2))
    c4.metric("Median", round(df["Amount"].median(),2))

    st.markdown("---")

    st.markdown("### Correlation Heatmap")

    corr = df.corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Feature Distribution")

    feature = st.selectbox(
        "Select Feature",
        df.columns[:-1]
    )

    fig = px.histogram(
        df,
        x=feature,
        nbins=50,
        title=f"{feature} Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)
# ----------------------------------------------------
# VISUALIZATIONS PAGE
# ----------------------------------------------------

elif page == "📈 Visualizations":

    st.title("📈 Interactive Visualizations")

    # -------------------------
    # Box Plot
    # -------------------------

    st.subheader("Transaction Amount Box Plot")

    fig = px.box(
        df,
        y="Amount",
        points="outliers",
        title="Amount Box Plot"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # Scatter Plot
    # -------------------------

    st.subheader("Scatter Plot")

    x_feature = st.selectbox(
        "Select X Axis",
        df.columns[:-1],
        key="x_feature"
    )

    y_feature = st.selectbox(
        "Select Y Axis",
        df.columns[:-1],
        index=1,
        key="y_feature"
    )

    sample_df = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    fig = px.scatter(
        sample_df,
        x=x_feature,
        y=y_feature,
        color=sample_df["Class"].astype(str),
        opacity=0.7,
        title="Feature Scatter Plot"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # Correlation with Target
    # -------------------------

    st.subheader("Feature Correlation with Fraud")

    corr = df.corr(numeric_only=True)["Class"].sort_values()

    fig = px.bar(
        x=corr.index,
        y=corr.values,
        labels={
            "x": "Features",
            "y": "Correlation"
        },
        title="Feature Correlation"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # Amount by Class
    # -------------------------

    st.subheader("Amount by Transaction Type")

    fig = px.box(
        df,
        x=df["Class"].replace({
            0: "Normal",
            1: "Fraud"
        }),
        y="Amount",
        color=df["Class"].replace({
            0: "Normal",
            1: "Fraud"
        }),
        title="Amount Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # Feature Explorer
    # -------------------------

    st.subheader("Feature Explorer")

    feature = st.selectbox(
        "Choose Feature",
        df.columns[:-1],
        key="feature_hist"
    )

    fig = px.histogram(
        df,
        x=feature,
        color=df["Class"].astype(str),
        nbins=50,
        marginal="box",
        title=f"{feature} Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # Correlation Matrix
    # -------------------------

    st.subheader("Top Correlation Matrix")

    selected = [
        "Amount",
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "Class"
    ]

    fig = px.imshow(
        df[selected].corr(),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)
# ----------------------------------------------------
# PREDICTION PAGE
# ----------------------------------------------------

elif page == "🤖 Prediction":

    st.title("🤖 Credit Card Fraud Prediction")

    st.write("Enter the transaction details below to predict whether it is Fraud or Normal.")

    st.markdown("---")

    input_data = {}

    col1, col2 = st.columns(2)

    features = [col for col in df.columns if col != "Class"]

    for i, feature in enumerate(features):

        default_value = float(df[feature].median())

        if i % 2 == 0:
            with col1:
                input_data[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.6f",
                    key=feature
                )
        else:
            with col2:
                input_data[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.6f",
                    key=feature
                )

    st.markdown("---")

    if st.button("🚀 Predict Fraud", use_container_width=True):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        fraud_probability = probability[1] * 100
        normal_probability = probability[0] * 100

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠ Fraud Transaction Detected")

        else:

            st.success("✅ Normal Transaction")

        c1, c2 = st.columns(2)

        c1.metric(
            "Normal Probability",
            f"{normal_probability:.2f}%"
        )

        c2.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

        st.markdown("---")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Normal", "Fraud"],
                y=[normal_probability, fraud_probability]
            )
        )

        fig.update_layout(
            title="Prediction Confidence",
            yaxis_title="Probability (%)",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        result = pd.DataFrame({
            "Feature": input_df.columns,
            "Value": input_df.iloc[0].values
        })

        st.subheader("Input Transaction")

        st.dataframe(result, use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction Report",
            csv,
            file_name="prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )
# ----------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------

elif page == "ℹ About":

    st.title("ℹ About This Project")

    st.markdown("""
    ## 💳 Credit Card Fraud Detection System

    This project is developed using **Machine Learning** and **Streamlit**
    to identify fraudulent credit card transactions.

    The application provides:

    ✅ Interactive Dashboard

    ✅ Exploratory Data Analysis (EDA)

    ✅ Interactive Visualizations

    ✅ Real-Time Fraud Prediction

    ✅ Download Prediction Report

    ✅ User Friendly Interface
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📂 Dataset Information")

        st.info(f"""
Dataset Name : Credit Card Fraud Detection

Rows : {df.shape[0]:,}

Columns : {df.shape[1]}

Target Column : Class
""")

    with col2:

        st.subheader("🤖 Machine Learning Model")

        st.success("""
Algorithm Used :

• Random Forest Classifier

Train/Test Split : 80 : 20

Model File :

fraud_model.pkl
""")

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    tech = pd.DataFrame({

        "Technology":[

            "Python",
            "Pandas",
            "NumPy",
            "Plotly",
            "Scikit-Learn",
            "Joblib",
            "Streamlit"

        ]

    })

    st.dataframe(
        tech,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("📈 Dataset Summary")

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Transactions",
        f"{len(df):,}"
    )

    c2.metric(
        "Fraud Cases",
        int(df["Class"].sum())
    )

    c3.metric(
        "Normal Cases",
        int((df["Class"]==0).sum())
    )

    c4.metric(
        "Features",
        df.shape[1]-1
    )

    st.markdown("---")

    st.success("✔ Model Loaded Successfully")

    st.success("✔ Dashboard Ready")

    st.success("✔ Prediction Module Active")

    st.markdown("---")

    st.markdown(
        """
<div style='text-align:center;
padding:20px;
border-radius:10px;
background-color:#0F172A;
color:white;
font-size:18px;'>

💳 Credit Card Fraud Detection Dashboard

Developed using Python • Streamlit • Machine Learning

© 2026 All Rights Reserved

</div>
""",
        unsafe_allow_html=True
    )