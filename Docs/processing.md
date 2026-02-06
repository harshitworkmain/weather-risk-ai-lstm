# Processing Module Documentation

## Overview
The `Src/processing` module handles data transformation, cleaning, and preparation. It ensures that raw weather data is converted into the specific format required by the LSTM neural network.

## Components

### `preprocessor.py`
Functions for initial data cleaning.

- **`clean_data(df)`**:
  - Handles missing values using forward-fill (`ffill`) followed by backward-fill (`bfill`) to maintain time-series continuity.
  - Fills any remaining NaNs with 0.
  - Returns a clean Pandas DataFrame.

### `feature_engine.py`
Classes for advanced feature engineering and scaling.

#### `FeatureEngine` Class
- **Methods**:
  - `scale_data(df, is_training)`:
    - Uses `MinMaxScaler` to normalize features to the [0, 1] range.
    - Persists the scaler to `Models/scalers/scaler.pkl` during training to ensure consistent scaling during inference.
  - `create_sliding_window(data, look_back, target_col_idx)`:
    - Converts a 2D time-series dataset into a 3D supervised learning format structure: `(Samples, TimeSteps, Features)`.
    - `look_back`: The number of past days to use for prediction (default: 10).
    - `target_col_idx`: The index of the column to predict (future value).
  - `inverse_transform(data, target_col_idx)`:
    - Handles the re-scaling of model predictions back to original units (e.g., Celsius).
    - Intelligent handling of shape mismatches (predicting 1 feature vs scaling 6 features).

## Usage Example
```python
from Src.processing.feature_engine import FeatureEngine

engine = FeatureEngine(config)
df_scaled = engine.scale_data(df)
X, y = engine.create_sliding_window(df_scaled, look_back=10)
```
