
import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import time

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
        • ARIMA
        • Optuna Hyperparameter Tuning

        Expected runtime: 10-15 minutes
        """
    )

if predict:

    if mode == " Quick Prediction":

        endpoint = (
            f"https://stock-predictor-api-fg3i.onrender.com/quick_predict/{ticker}"
        )
        with st.spinner(
            "Generating prediction..."
        ):
            response = requests.get(endpoint)
            result = response.json()

    else:

        endpoint = (
            f"https://stock-predictor-api-fg3i.onrender.com/start_analysis/{ticker}"
        )
        response = requests.get(endpoint)
        job = response.json()
        job_id = job['job_id']
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        elapsed =0
        progress_map = {
            'Downloading data': 10,
            'Creating features': 20,
            'Preparing data': 40,
            'Finding best model parameters': 60,
            'Training best model': 80,
            'Completed': 100
        }
        while True:
            response = requests.get(f"https://stock-predictor-api-fg3i.onrender.com/job_status/{job_id}")
            status = response.json()
            if status['status'] == 'completed':
                result = status['result']
                progress_bar.progress(100)
                progress_placeholder.success("Analysis completed")
                break
            elif status['status'] == 'error':
                st.error(status['message'])
                st.stop()
            step = status['step']
            progress_bar.progress(progress_map.get(step,0))
            progress_placeholder.info(f"""
                                      Current stage: {step}, 
                                      Time elapsed: {elapsed} seconds
                                      """)
            time.sleep(5)
            elapsed += 5

    
    #st.write(result)
    try:
        df = yf.download(ticker, period="1y", progress=False)
        st.subheader(f"{ticker} price history")
        st.line_chart(df["Close"])
    except Exception:
            pass
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current price", round(result["Current price"], 2))
    col2.metric("Predicted price", round(result["Predicted price"], 2))
    col3.metric("Signal", result["Signal"])
    col4.metric("Expected return", round(result["Expected return"], 2))
    col5.metric("Confidence", round(result["Confidence"], 2))


    st.subheader("Model Info")
    st.write(f"Best model: {result['best_model']}")
    st.write(f"Validation RMSE: {result['best_rmse']}")
    if result['best_model'] != 'arima':
        importance = pd.DataFrame({"Feature": result["Feature Importances"].keys(),"Importance": result["Feature Importances"].values()
})
        importance = importance.sort_values("Importance",ascending=False
)
        st.subheader("Feature Importances")
        st.bar_chart(importance.set_index("Feature"))
    else:
        st.info("ARIMA model does not have feature importances")

    st.subheader("Predicted Price")
    future_date = df.index[-1] + pd.Timedelta(days=5)
    fig , ax = plt.subplots(figsize=(10,5))
    ax.plot (df.index, df["Close"], label = "Historical Price")
    ax.scatter(future_date, result["Predicted price"], s=200, label = "Predicted Price")
    ax.legend()
    st.pyplot(fig)


