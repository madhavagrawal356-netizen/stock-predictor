
import streamlit as st
import requests

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Predictor")

st.write(
    "Predict stock movements using machine learning models."
)

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

mode = st.radio(
    "Prediction Mode",
    [
        " Quick Prediction",
        " Advanced AI Analysis"
    ]
)

if mode == " Quick Prediction":

    st.success(
        "Expected runtime: 5-15 seconds"
    )

else:

    st.warning(
        """
        Advanced AI Analysis

        • Random Forest
        • XGBoost
        • LightGBM
        • Optuna Hyperparameter Tuning

        Expected runtime: 10-15 minutes
        """
    )

if st.button("Predict"):

    if mode == " Quick Prediction":

        endpoint = (
            f"https://stock-predictor-api-fg3i.onrender.com/quick_predict/{ticker}"
        )

    else:

        endpoint = (
            f"https://stock-predictor-api-fg3i.onrender.com/deep_predict/{ticker}"
        )

    with st.spinner(
        "Generating prediction..."
    ):

        response = requests.get(endpoint)

        result = response.json()

    st.json(result)


