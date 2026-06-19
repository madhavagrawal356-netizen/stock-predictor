
from main_files.data_loader import load_stock
from main_files.features import create_features
from main_files.trainer import prepare_data
from main_files.optuna_model import get_model_category, tune_model, best_model, create_best_model
from main_files.evaluate import quick_evaluate
from sklearn.metrics import accuracy_score

# %%
def deep_predict(ticker):
    df = load_stock(ticker)
    df = create_features(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    best_type, best_params, best_rmse = best_model(X_train, y_train, X_test, y_test)
    winner = create_best_model(best_type, best_params)
    winner.fit(X_train, y_train)
    latest = X_test.iloc[[-1]]
    prediction = winner.predict(latest)[0]
    signal = 'Bearish' if prediction <0 else 'Bullish'
    y_pred_model = winner.predict(X_test)
    actual_direction = (y_test > 0).astype(int)
    predicted_direction = (y_pred_model > 0).astype(int)
    directional_accuracy = accuracy_score(actual_direction, predicted_direction)
    directional_accuracy = round(directional_accuracy*100, 2)
    return {
        "Ticker": ticker,
        "best_model": best_type,
        "best_rmse": best_rmse,
        "Expected return": float(100*prediction),
        "Confidence": directional_accuracy,
        "Signal": signal
    }


# %%
def quick_predict(ticker):
    model, best_rmse, model_name, X_test, y_test = quick_evaluate(ticker)
    latest = X_test.iloc[[-1]]
    prediction = model.predict(latest)[0]
    signal = 'Bearish' if prediction < 0 else 'Bullish'
    y_pred_model = model.predict(X_test)
    actual_direction = (y_test > 0).astype(int)
    predicted_direction = (y_pred_model > 0).astype(int)
    directional_accuracy = accuracy_score(actual_direction, predicted_direction)
    directional_accuracy = round(directional_accuracy*100, 2)
    return {
        "Ticker": ticker,
        "best_model": model_name,
        "best_rmse": best_rmse,
        "prediction": float(prediction),
        "signal": signal,
        "Confidence": directional_accuracy
    }


