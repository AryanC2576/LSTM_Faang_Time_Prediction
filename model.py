# model.py (Updated for 5-Layer LSTM + Attention + Dense + Dropout)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, GlobalAveragePooling1D, Dropout
import tensorflow as tf

def build_attention_lstm_model(input_shape, num_outputs):
    """
    Builds a deep LSTM model (5 layers) with an Attention mechanism,
    """
    input_layer = Input(shape=input_shape) # Input shape (lookback_window, num_features_per_timestep)

    # --- 5-Layer LSTM Stack ---
    # All LSTM layers return sequences, as the Attention layer needs the full sequence context.

    # Layer 1
    lstm_out = LSTM(units=128, return_sequences=True)(input_layer)
    # Optional: Add Dropout here for regularization between LSTM layers
    # lstm_out = Dropout(0.2)(lstm_out)

    # Layer 2
    lstm_out = LSTM(units=128, return_sequences=True)(lstm_out)
    # Optional: Add Dropout here
    # lstm_out = Dropout(0.2)(lstm_out)

    # Layer 3
    lstm_out = LSTM(units=128, return_sequences=True)(lstm_out)
    # Optional: Add Dropout here
    # lstm_out = Dropout(0.2)(lstm_out)

    # Layer 4
    lstm_out = LSTM(units=128, return_sequences=True)(lstm_out)
    # Optional: Add Dropout here
    # lstm_out = Dropout(0.2)(lstm_out)

    # Layer 5 (final LSTM layer before Attention)
    final_lstm_out = LSTM(units=128, return_sequences=True)(lstm_out)

    # --- Attention Mechanism ---
    # The Attention layer processes the full sequence output from the last LSTM layer.
    # This is a "self-attention" mechanism.
    # Correctly capture the single output tensor from the Attention layer
    attention_output = Attention()([final_lstm_out, final_lstm_out])

    # GlobalAveragePooling1D to convert the attention-weighted sequence
    # into a fixed-size vector summarizing the most important information.
    combined_context = GlobalAveragePooling1D()(attention_output)

    # --- Dense Layer after Attention ---
    # A standard Dense layer to process the rich context vector from Attention
    dense_layer_output = Dense(units=64, activation='relu')(combined_context)

    # --- Dropout Layer for Regularization ---
    # Helps prevent overfitting by randomly setting a fraction of input units to 0 at each update step during training.
    dropout_layer_output = Dropout(0.3)(dense_layer_output) # 0.3 is a common dropout rate, can be tuned

    # --- Multi-Output Heads ---
    # These output layers will take the processed features from the Dropout layer

    # Output head for price predictions
    price_output = Dense(num_outputs, activation='linear', name='price_output')(dropout_layer_output)

    # Output head for volatility predictions
    # Using 'softplus' for volatility to ensure positive output (volatility cannot be negative).
    volatility_output = Dense(num_outputs, activation='softplus', name='volatility_output')(dropout_layer_output)

    model = Model(inputs=input_layer, outputs=[price_output, volatility_output])

    model.compile(
        optimizer='adam',
        loss={'price_output': 'mae', 'volatility_output': 'mae'}, # Mean Absolute Error for both
        metrics={'price_output': ['mae'], 'volatility_output': ['mae']}
    )

    return model
