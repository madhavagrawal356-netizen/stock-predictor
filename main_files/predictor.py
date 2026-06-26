
from .data_loader import load_stock
from .features import create_features
from .trainer import prepare_data
from .optuna_model import get_model_category, tune_model, best_model, create_best_model
from .evaluate import quick_evaluate
from .job import update_job
from sklearn.metrics import accuracy_score
import os
import joblib
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# %%
Model_dir  ='models'
os.makedirs(Model_dir, exist_ok=True)
def deep_predict(ticker, job_id=None):
    model_path = (f"{Model_dir}/{ticker}.pkl")
    df = load_stock(ticker)
    if job_id:
        update_job(job_id, "running", "Creating features")
    df = create_features(df)
    if job_id:
        update_job(job_id, "running", "Preparing data")
    X_train, X_test, y_train, y_test = prepare_data(df)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        winner = model['model']
        best_rmse = model['best_rmse']
        best_type = model['best_type']
        best_params = model['best_params']
    else:
        if job_id:
            update_job(job_id, "running", "Finding best model parameters. This may take a while if the ticker is not in the backend.")
        best_type, best_params, best_rmse = best_model(X_train, y_train, X_test, y_test)
        if best_type == 'arima':
            if job_id:
                update_job(job_id, "running", "Training best model")
            winner = ARIMA(y_train, order=(best_params['p'], best_params['d'], best_params['q'])).fit()
            
        else:
            if job_id:
                update_job(job_id, "running", "Training best model")
            winner = create_best_model(best_type, best_params)
            winner.fit(X_train, y_train)
        joblib.dump({'model': winner, 'best_rmse': best_rmse, 'best_type': best_type, 'best_params': best_params, 'timestamp': datetime.now()}, model_path)
    latest = X_test.iloc[[-1]]
    if best_type == 'arima':
        prediction = winner.forecast(steps=1).iloc[0]
    else:
        prediction = winner.predict(latest)[0]
    signal = 'Bearish' if prediction <0 else 'Bullish'
    if best_type == 'arima':
        y_pred_model = winner.forecast(steps=len(y_test))
    else:
        y_pred_model = winner.predict(X_test)
    actual_direction = (y_test > 0).astype(int)
    predicted_direction = (y_pred_model > 0).astype(int)
    directional_accuracy = accuracy_score(actual_direction, predicted_direction)
    directional_accuracy = round(directional_accuracy*100, 2)
    predicted_price = float(df['Close'].iloc[-1])*(1+prediction)
    feature_importances = {}
    if best_type == 'arima':
        feature_importances = {}
    else:
        feature_importances = {feature: float(importance) for feature, importance in zip(X_train.columns, winner.feature_importances_)}
    return {
        "Ticker": ticker,
        "best_model": best_type,
        "best_rmse": best_rmse,
        "Expected return": float(100*prediction),
        "Confidence": directional_accuracy,
        "Signal": signal,
        "Predicted price": predicted_price,
        "Current price": float(df['Close'].iloc[-1]),
        "Feature Importances": feature_importances
    }


# %%
def quick_predict(ticker):
    model, best_rmse, model_name, X_test, y_test, df = quick_evaluate(ticker)
    latest = X_test.iloc[[-1]]
    prediction = model.predict(latest)[0]
    signal = 'Bearish' if prediction < 0 else 'Bullish'
    y_pred_model = model.predict(X_test)
    actual_direction = (y_test > 0).astype(int)
    predicted_direction = (y_pred_model > 0).astype(int)
    directional_accuracy = accuracy_score(actual_direction, predicted_direction)
    directional_accuracy = round(directional_accuracy*100, 2)
    predicted_price = float(df['Close'].iloc[-1])*(1+prediction)
    feature_importances={}
    feature_importances={feature : float(importance) for feature, importance in zip(X_test.columns, model.feature_importances_)}
    return {
        "Ticker": ticker,
        "best_model": model_name,
        "best_rmse": best_rmse,
        "Expected return": float(100*prediction),
        "Signal": signal,
        "Confidence": directional_accuracy,
        "Predicted price": predicted_price,
        "Current price": float(df['Close'].iloc[-1]),
        "Feature Importances": feature_importances
    }


# %% [markdown]
# joblib.dump({'model': winner, 'best_rmse': best_rmse, 'best_type': best_type, 'best_params': best_params, 'timestamp': datetime.now()}, model_path)

# %%
#deep_predict('AAPL')

# %% [markdown]
# joblib.dump({'model': winner, 'best_rmse': best_rmse, 'best_type': best_type, 'best_params': best_params, 'timestamp': datetime.now()}, model_path)

# %% [markdown]
# 


