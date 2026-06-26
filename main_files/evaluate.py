
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from .trainer import train_xgb, train_lgb , prepare_data, train_rf
from .data_loader import load_stock

# %%
from .features import create_features


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
    rmse_lgb, mae_lgb, r2_lgb = evaluate(train_lgb(X_train, y_train), X_test, y_test)
    rmse_rf, mae_rf, r2_rf = evaluate(train_rf(X_train, y_train), X_test, y_test)
    lowest_rmse = min([rmse_xgb, rmse_lgb, rmse_rf])
    best_model = train_xgb(X_train, y_train) if lowest_rmse == rmse_xgb else train_lgb(X_train, y_train) if lowest_rmse == rmse_lgb else train_rf(X_train, y_train)
    model_name = best_model.__class__.__name__
    return best_model , lowest_rmse, model_name, X_test, y_test, df2

# %%



