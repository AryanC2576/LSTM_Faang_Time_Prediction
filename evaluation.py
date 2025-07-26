
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(actuals_df, predictions_df):
    """
    Calculates performance metrics for each asset.
    """
    metrics = {}
    for col in actuals_df.columns:
        actuals = actuals_df[col].values
        
        predictions = predictions_df[col].values

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))

        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(actuals))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction[1:] == pred_direction[1:]) * 100

        metrics[col] = {'MAE': mae, 'RMSE': rmse, 'Directional Accuracy (%)': directional_accuracy}

    return pd.DataFrame(metrics).transpose()

def calculate_strategy_metrics(actuals_df, predictions_df, trading_days=252):

    metrics = {}
    for col in actuals_df.columns:
        # Calculate daily returns of the actual prices
        actual_returns = actuals_df[col].pct_change().dropna()

        last_known_prices = actuals_df[col].shift(1).dropna()
        aligned_predictions = predictions_df[col].loc[last_known_prices.index]

        # Create a signal: 1 for long (if predicted price > last known price), 0 otherwise.
        signal = (aligned_predictions > last_known_prices).astype(int)

        # Calculate the strategy's returns
        strategy_returns = actual_returns.loc[signal.index] * signal

        # --- Calculate Key Metrics ---

        # 1. Annualized Return
        annual_return = strategy_returns.mean() * trading_days

        # 2. Annualized Volatility (Standard Deviation)
        annual_volatility = strategy_returns.std() * np.sqrt(trading_days)

        # 3. Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan

        # 4. Maximum Drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        metrics[col] = {
            'Annual Return': annual_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

    return pd.DataFrame(metrics).transpose()

def plot_results(actuals_df, predictions_df,figname="results"):
    """
    Plots actual vs. predicted prices for each asset.
    """
    num_assets = actuals_df.shape[1]
    fig, axes = plt.subplots(num_assets, 1, figsize=(15, 5 * num_assets), sharex=True)

    if num_assets == 1:
        axes = [axes] # Ensure axes is an iterable for single asset case

    for i, col in enumerate(actuals_df.columns):
        axes[i].plot(actuals_df[col].index, actuals_df[col], label='Actual Price', color='blue')
        axes[i].plot(predictions_df[col].index, predictions_df[col], label='Predicted Price', color='red', linestyle='--')
        axes[i].set_title(f'Actual vs. Predicted Prices for {col.split("_")[1]}')
        axes[i].set_ylabel('Price')
        axes[i].legend()
        axes[i].grid(True)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f"{figname}.png") 
    plt.show()

def plot_performance_table(metrics_df, title="Performance Metrics by Asset"):
    """
    Plots a table of performance metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    tbl = ax.table(cellText=metrics_df.round(3).values,
                   colLabels=metrics_df.columns,
                   rowLabels=metrics_df.index,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.5)
    plt.title(title)
    plt.savefig(f"{title}.png") 
    plt.show()
