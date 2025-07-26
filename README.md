# LSTM FAANG Stock Time Series Prediction

This project implements a comprehensive framework for forecasting the "Adjusted Close" prices and volatility of FAANG (Meta/Facebook, Apple, Amazon, Netflix, and Google/Alphabet) and other selected stocks using a deep Long Short-Term Memory (LSTM) neural network with an Attention mechanism. It features a modular design for data preprocessing, feature engineering, model building, training, and performance evaluation through a rolling window forecasting approach.

The project analyzes stock data from 2020 to 2025, demonstrating an end-to-end machine learning pipeline for financial time series forecasting.

## Introduction

Accurately predicting stock prices is a challenging yet crucial task in financial markets. This project leverages the power of LSTMs, which are particularly effective at capturing temporal dependencies in sequence data, combined with an Attention mechanism to focus on the most relevant parts of the input sequence. The rolling window forecasting approach simulates a real-world scenario, where the model is continuously retrained and evaluated on new data.

## Features

> Modular Design: Code is organized into separate Python modules for data preprocessing, feature engineering, model definition, training/prediction, and evaluation.

> Historical Data Download: Automatically fetches historical stock data for specified tickers (e.g., AAPL, MSFT, GOOG) from Yahoo Finance.

> Comprehensive Data Preprocessing: Includes data flattening, handling of "Adj Close" and "Volume" data, and calculation of 5-day rolling volatility as a feature and a target.

> Feature Engineering: Calculates common technical indicators like Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD) to enrich the dataset.

> Advanced LSTM Model: Implements a 5-layer LSTM model incorporating an Attention mechanism for improved focus on relevant time steps, and Dropout for regularization.

> Multi-Output Prediction: The model is designed to predict both "Adjusted Close" prices and volatility simultaneously.

> Rolling Window Forecasting: Utilizes a robust rolling window strategy for training and prediction, simulating continuous forecasting with updated models.

> Detailed Performance Evaluation: Calculates key forecasting metrics (MAE, RMSE, Directional Accuracy) and financial strategy metrics (Annual Return, Sharpe Ratio, Max Drawdown).

> Visualization Tools: Provides functions to plot actual vs. predicted prices and display performance metrics in a clear tabular format.

## Technologies Used

> Python 3.10.12.

> pandas: For data manipulation and analysis

> numpy: For numerical operations

> yfinance: For downloading historical stock data

> scikit-learn: For data normalization (MinMaxScaler) and performance metrics (mean_absolute_error, mean_squared_error)

> tensorflow: For building and training the LSTM neural network

> matplotlib: For plotting and visualization

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

Ensure you have Python installed. The project was developed with Python 3.10.12

### Installation

#### Clone the repository:

Bash:

git clone https://github.com/AryanC2576/LSTM_Faang_Time_Prediction.git

#### Navigate into the project directory:

Bash:

cd LSTM_Faang_Time_Prediction

#### Install the required Python packages:
(It is recommended to create a virtual environment first)

Bash:

pip install -r requirements.txt

## Data

The data_preprocessing.py module automatically downloads historical stock data for the specified tickers from Yahoo Finance within the defined date range. No manual data download is required

## Project Structure
The repository is organized into several modular Python scripts and a Jupyter Notebook that orchestrates the workflow:

> faang_forecasting_project.ipynb: The main Jupyter Notebook that integrates all modules, defines configuration, and executes the end-to-end forecasting pipeline.

> data_preprocessing.py: Handles loading historical stock data, normalizing features using MinMaxScaler, and creating sequential data for LSTM input. It also calculates initial volatility features.

> feature_engineering.py: Contains functions to compute technical indicators like Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD) from the stock data.

> model.py: Defines the architecture of the Attention LSTM model, including its layers, input/output shapes, and compilation settings.

> training_and_prediction.py: Manages the rolling window forecasting process, including training the model on historical data and generating predictions for future time steps.

> evaluation.py: Provides functions to calculate various performance metrics for both the forecasting accuracy and the simulated trading strategy, and includes plotting utilities.

## Usage

The primary way to run this project is through the faang_forecasting_project.ipynb Jupyter Notebook.

### 1.Open the Jupyter Notebook:

Bash:

jupyter notebook faang_forecasting_project.ipynb

### 2.Execute cells sequentially:

The notebook is structured to run the entire pipeline step-by-step:

Configuration: Defines tickers (AAPL, MSFT, GOOG), date range (2020-01-01 to 2025-01-01), lookback_window (60 days), forecast_horizon (1 day), and train_ratio (0.8).

Data Loading: Calls load_data to download and prepare raw stock data.

Feature Engineering: Applies create_all_features to generate RSI and MACD.

Rolling Window Forecasting: Initiates the run_rolling_forecasts process, which iteratively trains the LSTM model and makes predictions.

Model Evaluation: Calculates and displays forecasting accuracy metrics using calculate_metrics.

Strategy Evaluation: Calculates and displays profitability metrics for a basic trading strategy using calculate_strategy_metrics.

Plotting Results: Generates visualizations of actual vs. predicted prices and performance tables using plot_results and plot_performance_table.

## Model Architecture

The core of the forecasting system is a deep LSTM model defined in model.py.

It consists of:

> Input Layer: Takes sequential data with a shape of (lookback_window, num_features_per_timestep).

> 5 LSTM Layers: Stacked LSTM layers, each with 128 units and configured to return_sequences=True to provide full sequence context to the subsequent Attention layer.

> Attention Mechanism: A self-attention layer processes the output of the final LSTM layer, allowing the model to weigh the importance of different time steps in the input sequence.

> Global Average Pooling: Converts the attention-weighted sequence into a fixed-size context vector.

> Dense Layer: A fully connected layer with 64 units and ReLU activation to process the combined context from the attention mechanism.

> Dropout Layer: Applied with a rate of 0.3 for regularization to prevent overfitting.

> Multi-Output Heads:

price_output: A Dense layer with num_outputs (equal to number of assets) and linear activation for predicting adjusted close prices.

volatility_output: A Dense layer with num_outputs and softplus activation (to ensure positive values) for predicting volatility.

The model is compiled with the Adam optimizer and uses Mean Absolute Error (MAE) as the loss function for both price and volatility outputs.

## Evaluation Metrics
The evaluation.py module provides functions to assess the model's performance from both a forecasting accuracy perspective and a simulated trading strategy perspective.

### Forecasting Model Performance
Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.

Root Mean Squared Error (RMSE): Measures the square root of the average of the squared errors, giving more weight to larger errors.

Directional Accuracy (%): Calculates the percentage of times the model correctly predicted the direction of the price movement (up or down).

### Trading Strategy Profitability
Annual Return: The annualized average return generated by the simple trading strategy.

Sharpe Ratio: Measures risk-adjusted return, indicating the return earned per unit of risk (volatility).

Max Drawdown: The maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained. It's a measure of downside risk.

## Visualizations
The evaluation.py module also includes functions to visualize the results:

> Actual vs. Predicted Prices: Plots comparing the true historical prices against the model's predictions for each asset.

> Performance Tables: Displays the calculated forecasting and strategy metrics in a clear, formatted table.

## Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1. Fork the repository.

2. Create a new branch (git checkout -b feature/AmazingFeature).

3. Commit your changes (git commit -m 'Add some AmazingFeature').

4. Push to the branch (git push origin feature/AmazingFeature).

5. Open a Pull Request.

## License

MIT License

## Contact

Aryan Chakravorty -:

Github Profile link: https://github.com/AryanC2576

Email-id: aryanchakravortu@gmail.com

Project Link: https://github.com/AryanC2576/LSTM_Faang_Time_Prediction
