**AI Stock Predictor**

<img width="947" height="443" alt="Screenshot 2026-07-02 013803" src="https://github.com/user-attachments/assets/2a6a2224-d1bc-4766-8305-5a51da1cedf8" />



An end-to-end stock forecasting application that predicts the expected 5-day return of publicly traded stocks using machine learning and statistical models. The project combines a FastAPI backend with a Streamlit dashboard to provide both fast predictions and more comprehensive AI-based analysis. The application was built to explore different approaches to time-series forecasting while providing a simple interface.

**Features**

1. Predicts the expected 5-day stock return
2. Two prediction modes: Quick Prediction (5–15 seconds), Advanced AI Analysis (10–15 minutes)
3. Automatic feature engineering using technical indicators
4. Compares multiple models and automatically selects the best performer
5. Hyperparameter optimization using Optuna
6. Model caching with Joblib to avoid retraining previously analyzed stocks
7. Scheduled Background checks and automatic model retraining for cached models for up to date predictions 
8. REST API built with FastAPI
9. Interactive dashboard built with Streamlit
10. Directional confidence score
11. Feature importance visualization for tree-based models

**Prediction Modes**
1. Quick Prediction :
Designed for fast inference by comparing:
XGBoost
Random Forest
ARIMA
The model with the lowest validation RMSE is selected automatically.
Typical runtime: 5–15 seconds
2.Advanced AI Analysis
Designed for maximum predictive performance.
Performs Optuna hyperparameter optimization
Compares multiple candidate models
Selects the best-performing model automatically
Saves trained models for future reuse
Typical runtime: 10–15 minutes (only the first time a ticker is analyzed)

**Feature Engineering**

The models are trained using both price information and technical indicators, including:

Closing Price
Trading Volume
Daily Return
5-Day Return
10-Day Return
Volatility
Momentum
Volume Change
RSI
MACD
MACD Signal
Lagged Returns

The prediction target is the expected percentage return over the next five trading days.



