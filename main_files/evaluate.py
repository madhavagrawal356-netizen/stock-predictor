
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from .trainer import train_xgb, prepare_data, train_rf
from .data_loader import load_stock

# %%
from .features import create_features
from statsmodels.tsa.arima.model import ARIMA


# %%
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2


# %%
def quick_evaluate(ticker):
    df = load_stock(ticker)
    df2 = df
    df = create_features(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    rmse_xgb, mae_xgb, r2_xgb = evaluate(train_xgb(X_train, y_train), X_test, y_test)
    
    rmse_rf, mae_rf, r2_rf = evaluate(train_rf(X_train, y_train), X_test, y_test)
    arima_model = ARIMA(y_train, order=(5,1,0)).fit()
    arima_pred = arima_model.forecast(steps=len(y_test))
    rmse_arima = np.sqrt(mean_squared_error(y_test, arima_pred))
    mae_arima = mean_absolute_error(y_test, arima_pred)
    r2_arima = r2_score(y_test, arima_pred)
    lowest_rmse = min([rmse_xgb, rmse_arima, rmse_rf])
    if lowest_rmse == rmse_xgb:
        best_model = train_xgb(X_train, y_train)
        model_name = 'XGBoost'
    elif lowest_rmse==rmse_rf:
        best_model = train_rf(X_train, y_train)
        model_name = 'RandomForest'
    else:
        best_model=arima_model
        model_name='ARIMA'
    return best_model , lowest_rmse, model_name, X_test, y_test, df2

# %%



