import pandas as pd
def calculate_rsi(data_frame, period=14):
    """
    Calculates the Relative Strength Index (RSI) for each asset.
    """
    for col in data_frame.columns:
        if 'Adj Close' in col:
            delta = data_frame[col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            data_frame[f"RSI_{col.split('_')[1]}"] = 100 - (100 / (1 + rs))
    return data_frame
"""
We can also do this by using the ta library for performance and accuracy
from ta.momentum import RSIIndicator
def calculate_rsi(data_frame, period=14):
    print(f"Calculating RSI for period {period} using TA-Lib...")
    df_copy = data_frame.copy() # Work on a copy to avoid modifying original in place prematurely
    
    for col in df_copy.columns:
        if 'Adj Close' in col:
            ticker = col.split('_')[1]
            # TA-Lib RSI function expects a Series or NumPy array
            df_copy[f"RSI_{ticker}"] = RSIIndicator(df_copy[col].values, timeperiod=period)
            
    return df_copy
"""
def calculate_macd(data_frame, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD) for each asset.
    """
    print(f"Calculating MACD (Fast:{fast_period}, Slow:{slow_period}, Signal:{signal_period})...")
    for col in data_frame.columns:
        if 'Adj Close' in col:
            # Calculate the Fast EMA
            exp1 = data_frame[col].ewm(span=fast_period, adjust=False).mean()
            # Calculate the Slow EMA
            exp2 = data_frame[col].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate the MACD line (Fast EMA - Slow EMA)
            macd = exp1 - exp2
            # Calculate the Signal line (EMA of the MACD line)
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            
            data_frame[f"MACD_{col.split('_')[1]}"] = macd
            data_frame[f"MACD_Signal_{col.split('_')[1]}"] = signal
            
    return data_frame

"""
We can also do this by using the ta library for performance and accuracy
from ta.trend import MACD
def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    macd = MACD(close=df['Close'], window_slow=slow_period, window_fast=fast_period, window_sign=signal_period)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    return df
"""
def create_all_features(data_frame):
    print("Creating all features...")
    # Calculate RSI
    df_with_rsi = calculate_rsi(data_frame.copy()) # Use a copy to avoid modifying original df directly if passed from other modules

    # Calculate MACD (on the DataFrame already containing RSI)
    df_with_macd = calculate_macd(df_with_rsi.copy())

    # Drop any rows that contain NaN values, which arise from the initial
    # periods of rolling window calculations (e.g., first 14 days for RSI).
    # This ensures a clean dataset for the LSTM.
    final_df = df_with_macd.dropna()
    print(f"Features created. Original data rows: {len(data_frame)}, Rows after feature engineering: {len(final_df)}")
    return final_df
