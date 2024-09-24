import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import logging
import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

def get_data(symbol, start_date, end_date):
    # Use a financial API to fetch historical price and volume data
    # Example using Alpha Vantage:
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey=YOUR_API_KEY&outputsize=full"
    data = pd.read_json(url)
    # Clean and preprocess the data
    df = pd.DataFrame(data["Time Series (Daily)"])
    df.index = pd.to_datetime(df.index)
    df = df.T.reset_index().rename(columns={"index": "Date"})

    # Handle missing values and outliers
    df.dropna(inplace=True)
    # Add outlier detection and handling if necessary

    return df

# 2. Feature Engineering
def create_features(df):
    # Create a wider range of features
    df["SMA_50"] = df["5. adjusted close"].rolling(window=50).mean()
    df["SMA_200"] = df["5. adjusted close"].rolling(window=200).mean()
    df["RSI"] = ta.rsi(df["5. adjusted close"], timeperiod=14)
    df["MACD"] = ta.macd(df["5. adjusted close"], fastperiod=12, slowperiod=26, signalperiod=9)
    # Add more features based on technical indicators, volatility measures, etc.
    return df

# 3. Model Selection and Training
def train_model(df):
    try:
        # Split data into features and target variable
        X = df.drop("5. adjusted close", axis=1)
        y = df["5. adjusted close"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Use a more complex model and hyperparameter tuning
        model = RandomForestRegressor()
        param_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate model performance
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Mean Squared Error: {mse}")

        return best_model

    except Exception as e:
        logging.error(f"Error during model training: {e}")

# 4. Trading Strategy with Risk Management
def execute_trades(model, df, position_size=1000, stop_loss_pct=0.05):
    try:
        # Generate buy/sell signals based on model predictions and trading rules
        if model.predict(df.iloc[-1].values.reshape(1, -1)) > df["5. adjusted close"].iloc[-1]:
            signal = "buy"
        else:
            signal = "sell"

        # Log trading signals
        logging.info(f"Signal: {signal}")
        logging.info(f"Entry Price: {df['5. adjusted close'].iloc[-1]}")

        # Execute trades with risk management
        if signal == "buy":
            # Calculate stop-loss price
            stop_loss_price = df["5. adjusted close"].iloc[-1] * (1 - stop_loss_pct)
            # Execute buy order with position sizing
        elif signal == "sell":
            # ... (similar logic for selling)

    except Exception as e:
        logging.error(f"Error during trade execution: {e}")

# 5. Backtesting
def backtest(model, df):
    # Simulate trading using historical data to evaluate performance
    # ... (implement backtesting logic)

    # Calculate portfolio performance
    # ... (calculate profit/loss, ROI, etc.)
    logging.info(f"Portfolio Performance: {performance}")

# Main Execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    symbol = config['stock']['symbol']
    start_date = config['stock']['start_date']
    end_date = config['stock']['end_date']

    df = get_data(symbol, start_date, end_date)
    df = create_features(df)
    model = train_model(df)
    backtest(model, df)
