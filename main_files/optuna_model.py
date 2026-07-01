

import optuna
import numpy as np
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from .trainer import prepare_data
from .data_loader import load_stock
from .features import create_features
from sklearn.model_selection import TimeSeriesSplit
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore",ConvergenceWarning)

# %%
def get_model_category(trial, model_type):
    if model_type == "xgb":
        model = XGBRegressor(n_estimators = trial.suggest_int("n_estimators", 100, 1000), max_depth = trial.suggest_int("max_depth", 1, 15), learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1), subsample = trial.suggest_float("subsample", 0.5, 1.0), colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0), random_state = 42)
    elif model_type == "arima":
        return {"p": trial.suggest_int("p", 0, 5), "d": trial.suggest_int("d", 0, 2), "q": trial.suggest_int("q", 0, 5)}
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=trial.suggest_int("n_estimators", 100, 1000), max_depth=trial.suggest_int("max_depth", 1, 15), min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),min_samples_split=trial.suggest_int("min_samples_split", 2, 10), random_state=42)
    return model




# %%
#df = load_stock('MSFT')

# %%
#df = create_features(df)
#X_train, X_test, y_train, y_test = prepare_data(df)

# %%
def tune_model(model_type, X_train, y_train, X_test, y_test, n_trials=5):
    def objective(trial):
        

        tscv = TimeSeriesSplit(n_splits = 5)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]
            if model_type == "arima":
                try:
                    params = get_model_category(trial, model_type)
                    model = ARIMA(y_tr, order=(params['p'], params['d'], params['q']))
                    fitted = model.fit()
                    y_pred = fitted.forecast(steps = len(y_val))
                except Exception:
                    return float('inf')
            else:
                model = get_model_category(trial, model_type)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)

            scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        
        return np.mean(scores)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study


# %%
def best_model(X_train, y_train, X_test, y_test):
    studies={}
    studies['rf'] = tune_model('rf' , X_train, y_train, X_test, y_test)
    studies['xgb'] = tune_model('xgb', X_train, y_train, X_test, y_test)
    studies['arima'] = tune_model('arima', X_train, y_train, X_test, y_test)
    
    best_type = min(studies, key=lambda x: studies[x].best_value)
    best_rmse = studies[best_type].best_value
    best_params = studies[best_type].best_params
    return best_type, best_params, best_rmse
  

# %%
#best_type , best_params, best_rmse = best_model(X_train, y_train, X_test, y_test)
#print(best_params)
#print(best_rmse)
#print(best_type)

# %%
def create_best_model(best_type , best_params):
    if best_type == 'xgb':
        model = XGBRegressor(**best_params, random_state=42)
    elif best_type == 'arima':
        return best_params
    elif best_type == 'rf':
        model = RandomForestRegressor(**best_params, random_state=42)
    return model
#winner = create_best_model(best_type, best_params)
#winner.fit(X_train, y_train)
#y_pred = winner.predict(X_test)
#final_score = np.sqrt(mean_squared_error(y_test, y_pred))
#print(f'Final Score: {final_score}')



# %%




