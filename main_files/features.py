
def create_features(df):
    df = df.copy()
    df['Return']= df['Close'].pct_change()
    df['Return_5']= df['Close'].pct_change(5)
    df['Return_10']= df['Close'].pct_change(10)
    #df['MA5']=df['Close'].rolling(window=5).mean()
    #df['MA9']=df['Close'].rolling(window=9).mean()
    #df['MA20']=df['Close'].rolling(window=20).mean()
    df['Volatality']=df['Return'].rolling(window=20).std()
    df['Volume_change']=df['Volume'].pct_change()
    df['Momentum']=df['Close']/df['Close'].shift(10)-1
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = (
    df["MACD"].ewm(span=9, adjust=False).mean())
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Lag_1"] = df["Return"].shift(1)
    df["Lag_2"] = df["Return"].shift(2)
    df["Lag_3"] = df["Return"].shift(3)
    df["Lag_5"] = df["Return"].shift(5)
    df['Target']= df['Close'].shift(-5)/df['Close']-1
    return df.dropna()

# %%



