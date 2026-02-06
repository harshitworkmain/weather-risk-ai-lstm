# Modeling Module Documentation

## Overview
The `Src/modeling` module encapsulates the Deep Learning logic. It uses **TensorFlow/Keras** to build, train, and manage the LSTM (Long Short-Term Memory) models used for forecasting.

## Components

### `lstm_model.py`
Defines the neural network architecture.

- **`build_lstm_model(input_shape)`**:
  - **Layer 1**: `LSTM(50, return_sequences=True)` - Captures temporal patterns.
  - **Dropout**: `0.2` - Prevents overfitting.
  - **Layer 2**: `LSTM(50, return_sequences=False)` - Condenses temporal features.
  - **Dense Layer**: `Dense(25, activation='relu')` - Learn non-linear combinations.
  - **Output**: `Dense(1)` - Regression output (predicted value).
  - **Optimizer**: `Adam`.
  - **Loss Function**: `Mean Squared Error (MSE)`.

### `trainer.py`
Manages the training lifecycle.

#### `Trainer` Class
- **Methods**:
  - `train(X_train, y_train)`:
    - Orchestrates the model training loop.
    - Implements **Early Stopping** (patience=5) to stop training when validation loss stabilizes.
    - Implements **ModelCheckpoint** to automatically save the best weights to `Models/checkpoints/best_model.keras`.

## Configuration
Hyperparameters are controlled via `Config/config.yaml`:
```yaml
model:
  lstm:
    look_back: 10
    epochs: 20
    batch_size: 32
```
