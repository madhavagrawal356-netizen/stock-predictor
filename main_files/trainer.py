
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

# %%
def prepare_data(df):
    features = ['Momentum', 'Close', 'Volume', 'Return','Return_5','Return_10', 'Volatality', 'RSI', 'MACD', 'MACD_Signal', 'Volume_change', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
    X = df[features]
    y = df['Target']
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test
#'MA5', 'MA9', 'MA20',

# %%
def train_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators = 100, max_depth = 5, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8)
    model.fit(X_train, y_train)
    return model

def train_lgb(X_train, y_train):
    model = LGBMRegressor(n_estimators = 100)
    model.fit(X_train, y_train)
    return model

# %%
def train_rf(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# %%



