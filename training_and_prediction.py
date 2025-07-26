
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Assuming create_sequences is suitable for creating X and y pairs from a single window
from data_preprocessing import create_sequences
import numpy as np
from model import build_attention_lstm_model # Import model builder here if not passed in

def run_rolling_forecasts(data, model_builder, lookback_window, forecast_horizon, train_ratio=0.8):
    """
    Executes a rolling window forecast.
    For each step, it trains a model on a growing training window
    and forecasts the next `forecast_horizon` steps.
    """
    # Assume tickers list is available in the global scope or passed in
    # For now, let's assume it's implicitly available or we need to derive it
    # Let's derive it from the 'Adj Close' column names in the data
    price_cols = [col for col in data.columns if 'Adj Close' in col]
    # *** FIX: Calculate num_assets based on the number of price columns (which should equal the number of tickers) ***
    num_assets = len(price_cols) # This correctly represents the number of assets

    num_total_timesteps = len(data)
    # The first possible prediction point is after the initial training data + lookback window + forecast horizon
    # The loop should iterate over the points for which we want to make a prediction
    # The index i will represent the start of the forecast horizon for the current step
    start_of_forecast_index = int(num_total_timesteps * train_ratio)

    predictions = []
    actuals = []

    # Get column names for price and volatility targets from the *feature_df*
    # price_cols is already defined above

    volatility_cols = [col for col in data.columns if 'Volatility' in col]


    # Find the indices of the target columns in the *feature_df* array
    # These indices are used to slice y_train_full, which is based on feature_df
    price_indices = [data.columns.get_loc(col) for col in price_cols]
    volatility_indices = [data.columns.get_loc(col) for col in volatility_cols]

    print(f"num_assets: {num_assets}")
    print(f"len(price_indices): {len(price_indices)}")
    print(f"len(volatility_indices): {len(volatility_indices)}")


    # The rolling window will move one step at a time, making a forecast
    # The loop should go up to the point where we have enough data for the last forecast horizon
    # The index `i` represents the starting index of the forecast horizon for the current iteration.
    # The training data for iteration `i` will be up to `i - 1`.
    # The input sequence for the model will be `data[i - lookback_window : i]`
    # The target for the model will be `data[i : i + forecast_horizon]`
    for i in range(start_of_forecast_index, num_total_timesteps - forecast_horizon + 1, forecast_horizon):

        # Define the training data (all data up to the start of the current forecast)
        train_data_window = data.iloc[:i].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Normalize the training data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data_window)

        # Create sequences for training
        # X_train shape: (num_sequences, lookback_window, num_features)
        # y_train_full shape: (num_sequences, forecast_horizon, num_features)
        X_train, y_train_full = create_sequences(train_scaled, lookback_window, forecast_horizon)

        # Extract target variables for the multi-output model from the training sequences
        # Reshape y_train_full from (num_sequences, forecast_horizon, num_features)
        # to (num_sequences * forecast_horizon, num_target_features) for model training
        # Ensure y_train_full is not empty before slicing/reshaping
        if y_train_full.shape[0] == 0:
            print(f"Warning: No training sequences created for forecast starting at index {i}. Skipping training.")
            continue # Skip this forecast step if no training data

        # Slice based on the indices calculated from feature_df columns
        # *** FIX: Ensure reshape target size matches the number of selected columns ***
        y_train_price = y_train_full[:, :, price_indices].reshape(-1, len(price_indices))

        print(f"i: {i}")
        print(f"y_train_full shape: {y_train_full.shape}")
        print(f"y_train_full[:, :, volatility_indices].shape: {y_train_full[:, :, volatility_indices].shape}")
        print(f"num_assets: {num_assets}")
        print(f"len(volatility_indices): {len(volatility_indices)}")


        y_train_volatility = y_train_full[:, :, volatility_indices].reshape(-1, len(volatility_indices))


        # Define the input data for the prediction (the single lookback window right before the forecast)
        # This should be the last `lookback_window` data points from the training window
        input_sequence_data = train_data_window.iloc[-lookback_window:].copy() # Use .copy()

        # Normalize the input sequence using the SAME scaler fitted on the training data
        input_sequence_scaled = scaler.transform(input_sequence_data)

        # Reshape the input sequence for the model (add batch dimension)
        X_predict = np.array([input_sequence_scaled]) # Shape (1, lookback_window, num_features)


        # Build and train the model at each step (or load a pre-trained one)
        # Building and training at each step is computationally expensive but reflects the prompt's structure.
        # In a real-world scenario, you might train once and update, or use a fixed window.
        # *** FIX: Pass the correct number of outputs (num_assets) to the model builder ***
        model = build_attention_lstm_model((X_train.shape[1], X_train.shape[2]), num_assets)

        print(f"Training model for forecast starting at index {i}...")
        # Ensure there are sequences to train on
        if X_train.shape[0] > 0:
             # *** FIX: Pass the correct target shapes to model.fit ***
             model.fit(
                 X_train,
                 {'price_output': y_train_price, 'volatility_output': y_train_volatility},
                 epochs=5, batch_size=32, verbose=0
             )
        else:
             print(f"Warning: No training sequences created for forecast starting at index {i}. Skipping training.")
             continue # Skip this forecast step if no training data


        # Make a prediction using the single input sequence
        if X_predict.shape[0] > 0:
            # The model outputs a list of predictions, one for each output head
            predictions_scaled = model.predict(X_predict)
            price_prediction_scaled = predictions_scaled[0] # Shape (1, num_assets)
            # volatility_prediction_scaled = predictions_scaled[1] # Shape (1, num_assets) # Not used in evaluation metrics currently


            # Inverse transform the price prediction to get the actual price values
            # Create a placeholder array for inverse transformation with all features
            # This is necessary because the scaler was fitted on all features
            # The number of columns in the dummy array must match the number of columns in feature_df
            dummy_array = np.zeros((forecast_horizon, data.shape[1]))

            # Place the scaled price predictions into the correct columns of the dummy array
            # price_prediction_scaled has shape (1, num_assets), need to reshape to (forecast_horizon, num_assets)
            # Since forecast_horizon is 1, price_prediction_scaled is (1, num_assets).
            # We need to expand it to (1, num_assets) to fit into the dummy array row(s).
            # If forecast_horizon > 1, the model output shape would be (1, forecast_horizon * num_assets) or similar,
            # requiring a more complex reshaping here. Assuming forecast_horizon = 1 for now.
            if forecast_horizon == 1:
                 # Ensure the number of elements in price_prediction_scaled matches the number of price indices
                 if price_prediction_scaled.shape[1] == len(price_indices):
                     dummy_array[0, price_indices] = price_prediction_scaled[0, :] # Place the 1-step prediction
                 else:
                     print(f"Error: Mismatch between predicted price columns ({price_prediction_scaled.shape[1]}) and price indices ({len(price_indices)}).")
                     # Decide how to handle this - skipping prediction or raising error
                     continue # Skip this forecast step


            else:
                 # If forecast_horizon > 1, need to reshape price_prediction_scaled to (forecast_horizon, num_assets)
                 # This assumes the model's output layer for price is structured to output this way.
                 # The current model has Dense(num_outputs), which is Dense(num_assets), so it outputs (batch_size, num_assets).
                 # If forecast_horizon > 1, the model output should probably be Dense(forecast_horizon * num_assets)
                 # and then reshaped here. Let's assume forecast_horizon = 1 for this model structure.
                 print("Warning: Model output structure is designed for forecast_horizon=1. Inverse transforming for forecast_horizon > 1 might be incorrect.")
                 # Attempt to reshape if needed, but this needs careful checking against model output structure
                 # dummy_array[:, price_indices] = price_prediction_scaled.reshape(forecast_horizon, num_assets)
                 pass # Stick to forecast_horizon = 1 for now


            # Inverse transform the dummy array to get descaled predictions for the price columns
            # Slice the inverse transformed array using price_indices to get only the predicted price columns
            prediction_descaled = scaler.inverse_transform(dummy_array)[:, price_indices].flatten() # Flatten to 1D array of shape (num_assets,)


            # Get the actual values for comparison (the actual prices for the forecast horizon)
            # The actual data for the forecast horizon starts at index `i` and goes for `forecast_horizon` steps
            actual_data_window = data.iloc[i : i + forecast_horizon].copy() # Shape (forecast_horizon, num_features)
            actual_descaled = actual_data_window[price_cols].values.flatten() # Shape (forecast_horizon * num_assets,) -> (num_assets,) if forecast_horizon=1


            predictions.append(prediction_descaled)
            actuals.append(actual_descaled)

            print(f"Forecast step {i - start_of_forecast_index + 1}/{num_total_timesteps - start_of_forecast_index - forecast_horizon + 1} completed.")
        else:
             print(f"Warning: No prediction input sequence created for forecast starting at index {i}. Skipping prediction.")


    # Convert lists of predictions and actuals to DataFrames
    # Each row in the DataFrame should represent one forecast step, with columns for each asset.
    # The shape of predictions and actuals lists is (num_forecast_steps, num_assets)
    predictions_df = pd.DataFrame(predictions, columns=price_cols)
    actuals_df = pd.DataFrame(actuals, columns=price_cols)

    # Set the index of the results DataFrames to correspond to the start of the forecast horizon
    # The index should be the date at which the prediction is made, which is the start of the forecast window
    forecast_dates = data.index[start_of_forecast_index : num_total_timesteps - forecast_horizon + 1 : forecast_horizon]

    # Ensure the number of rows in predictions_df/actuals_df matches the number of forecast dates
    # This check is important if some forecast steps were skipped (e.g., due to no training data)
    if len(predictions_df) == len(forecast_dates):
         predictions_df.index = forecast_dates
         actuals_df.index = forecast_dates
    else:
         print(f"Warning: Number of forecast steps ({len(predictions_df)}) does not match the number of expected dates ({len(forecast_dates)}). Indexing might be incorrect.")
         # If mismatch, we might need a more robust way to track dates for successful forecasts


    return {'actuals': actuals_df, 'predictions': predictions_df}
