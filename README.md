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

**Why this is needed**

Most "stock predictor" projects are a single notebook that overfits one ticker. This one is built as a small system instead:
1. Per-ticker model selection — no assumption that one model architecture wins for every stock
2. Proper time-series validation — TimeSeriesSplit inside every Optuna trial, chronological train/test split, no shuffling

**Prediction Modes**
1. **Quick Prediction** :<br>
Designed for fast inference by comparing:<br>
XGBoost<br>
Random Forest<br>
ARIMA<br>
The model with the lowest validation RMSE is selected automatically.<br>
**Typical runtime: 5–15 seconds**

2. **Advanced AI Analysis**
Designed for maximum predictive performance.<br>
Performs Optuna hyperparameter optimization<br>
Compares multiple candidate models<br>
Selects the best-performing model automatically<br>
Saves trained models for future reuse<br>
**Typical runtime: 10–15 minutes (only the first time a ticker is analyzed)**<br>

**Feature Engineering**

The models are trained using both price information and technical indicators, including:

Closing Price<br>
Trading Volume<br>
Daily Return<br>
5-Day Return<br>
10-Day Return<br>
Volatility<br>
Momentum<br>
Volume Change<br>
RSI<br>
MACD<br>
MACD Signal<br>
Lagged Returns<br>

The prediction target is the expected percentage return over the next five trading days.

**Evaluation Metrics**

Models are evaluated using:

Root Mean Squared Error (RMSE)<br>
Mean Absolute Error (MAE)<br>
R² Score<br>
Directional Accuracy<br>

The model with the lowest validation RMSE is selected for prediction.

**Tech Stack**

| Layer | Tools |
|----------|----------|
| Modeling  | XGBoost, scikit-learn (Random Forest), statsmodels (ARIMA)  | 
| Tuning  | Optuna with TimeSeriesSplit, cross-validation  | 
| Data | yfinance|
|Backend| FastAPI, background threading, APScheduler|
|Frontend| Streamlit|
|Persistence|joblib (models), JSON (job state)|

**Project Structure**

```text
main_files/
├── data_loader.py      # yfinance pull
├── features.py         # Technical indicators + target construction
├── trainer.py          # Train/test split and model training
├── optuna_model.py     # Optuna search for each model type
├── evaluate.py         # Quick-mode model comparison
├── predictor.py        # quick_predict / deep_predict orchestration
├── job.py              # Async job status persistence
scheduler.py        # Background retraining every 60 days
app.py              # FastAPI routes
dashboard.py        # Streamlit dashboard

```

**Running Locally**<br>

pip install -r requirements.txt<br>

**backend**<br>
uvicorn app:app --reload<br>

**frontend (in a second terminal)** <br>
streamlit run dashboard.py
