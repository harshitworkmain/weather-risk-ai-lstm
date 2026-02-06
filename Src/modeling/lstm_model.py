from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    
    Args:
        input_shape (tuple): (timesteps, features)
        
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'), # Added activation based on typical patterns, check notebook if it had one.
        # Notebook summary said "Dense(25, activation='relu')" in my previous turn.
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
