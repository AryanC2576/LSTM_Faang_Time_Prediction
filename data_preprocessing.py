import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd # Ensure pandas is imported here
import numpy as np

def load_data(tickers, start_date, end_date):
    print("Downloading historical data...")
    try:
        # Set auto_adjust=False to get the 'Adj Close' column
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            raise ValueError("No data downloaded. Check tickers and dates.")

        # Flatten the multi-level column index
        if isinstance(data.columns, pd.MultiIndex):
             # Ensure the order of levels is (Attribute, Ticker)
             # yfinance can sometimes return (Ticker, Attribute), so reorder if necessary
             if data.columns.names[0] != 'Attributes':
                 data.columns = data.columns.swaplevel(0, 1)
                 data.columns.names = ['Attributes', 'Symbols'] # Standardize names if needed


             data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
        else:
             # For a single ticker, columns might just be ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
             if len(tickers) == 1:
                 ticker = tickers[0]
                 data.columns = [f"{col}_{ticker}" for col in data.columns]
             pass # Columns are already in a flat format, hope they are named correctly


        adj_close_cols = [col for col in data.columns if 'Adj Close' in col]
        volume_cols = [col for col in data.columns if 'Volume' in col]

        # Check if we got Adj Close data for all tickers
        if len(adj_close_cols) != len(tickers):
             print(f"Warning: Did not find 'Adj Close' data for all tickers. Expected {len(tickers)}, found {len(adj_close_cols)}")
             # For now, let's raise an error to be safe
             raise ValueError(f"Missing 'Adj Close' data for one or more tickers. Found columns: {adj_close_cols}")


        # We also need to compute volatility for our target
        combined_data = data[adj_close_cols + volume_cols].copy()

        # Calculate daily returns for volatility calculation
        returns_df = combined_data[adj_close_cols].pct_change().fillna(0)

        # Add print statements to debug the indexer error
        print("Debugging volatility calculation:")
        print(f"adj_close_cols: {adj_close_cols}")
        print(f"returns_df columns: {returns_df.columns.tolist()}")
        print(f"returns_df shape: {returns_df.shape}")

        # Calculate 5-day rolling standard deviation of returns as a volatility proxy
        volatility_cols_names = []
        for adj_close_col in adj_close_cols:
            ticker = adj_close_col.split('_')[1]
            vol_col_name = f"Volatility_{ticker}"
            volatility_cols_names.append(vol_col_name)
            # Calculate rolling std directly on the returns column
            if adj_close_col in returns_df.columns:
                combined_data[vol_col_name] = returns_df[adj_close_col].rolling(window=5).std()
            else:
                print(f"Error: Could not find returns column '{adj_close_col}' for ticker {ticker}. Skipping volatility calculation for this asset.")


        combined_data = combined_data.dropna()

        print("Data loaded successfully.")
        return combined_data

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def normalize_features(data_frame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_frame)
    return scaled_data, scaler

def create_sequences(data, lookback_window, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - lookback_window - forecast_horizon + 1):
        X.append(data[i:(i + lookback_window)])
        y.append(data[i + lookback_window : i + lookback_window + forecast_horizon])
    return np.array(X), np.array(y)
