
import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Stock Predictor",
    layout="wide"
)

st.title(" AI Stock Predictor")

st.caption(
    "Predict 5-day stock returns using machine learning models."
)

with st.sidebar:
    st.header("Select Stock and Prediction Type")
    ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL")
    mode = st.radio(
    "Prediction Mode",
    [
        " Quick Prediction",
        " Advanced AI Analysis"
    ])
    predict = st.button("Generate Forecast")

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

if predict:

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
    
    try:
        df = yf.download(ticker, period="1y", progress=False)
        st.subheader(f"{ticker} price history")
        st.line_chart(df["Close"])
    except Exception:
            pass
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current price", round(result["Current price"], 2))
    col2.metric("Predicted price", round(result["Predicted price"], 2))
    col3.metric("Signal", result["signal"])
    col4.metric("Expected return", round(result["Expected return"], 2))
    col5.metric("Confidence", round(result["Confidence"], 2))


    st.subheader("Model Info")
    st.write(f"Best model: {result['best_model']}")
    st.write(f"Validation RMSE: {result['best_rmse']}")
    importance = pd.DataFrame({
    "Feature": result["Feature Importances"].keys(),
    "Importance": result["Feature Importances"].values()
})
    importance = importance.sort_values(
    "Importance",
    ascending=False
)
    st.subheader("Feature Importances")
    st.bar_chart(importance.set_index("Feature"))

    st.subheader("Predicted Price")
    future_date = df.index[-1] + pd.Timedelta(days=5)
    fig , ax = plt.subplots(figsize=(10,5))
    ax.plot (df.index, df["Close"], label = "Historical Price")
    ax.scatter(future_date, result["Predicted price"], s=200, label = "Predicted Price")
    ax.legend()
    st.pyplot(fig)


