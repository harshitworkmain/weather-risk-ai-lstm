import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scalers_dir = self.config['data']['paths'].get('processed', 'Data/processed')

    def scale_data(self, df: pd.DataFrame, is_training: bool = True):
        """
        Scale features using MinMaxScaler.
        If training, fit then transform.
        If inference, load scaler (or use existing) and transform.
        """
        data = df.values
        
        if is_training:
            scaled_data = self.scaler.fit_transform(data)
            self.save_scaler()
        else:
            # If we don't have a fit scaler, try loading
            # For now, assuming caller handles correct flow or we reuse self.scaler
            scaled_data = self.scaler.transform(data)
            
        return pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

    def create_sliding_window(self, data: pd.DataFrame, look_back: int = None, target_col_idx: int = 0):
        """
        Create sliding window sequences for LSTM.
        X: (samples, time_steps, features)
        y: (samples, target)
        
        Args:
            data (pd.DataFrame): Scaled input data.
            look_back (int): Number of previous time steps to use.
            target_col_idx (int): Index of the column to predict (default 0).
        """
        if look_back is None:
            look_back = self.config['model']['lstm']['look_back']

        dataset = data.values
        X, y = [], []
        
        for i in range(len(dataset) - look_back):
            # Sequence of length 'look_back'
            a = dataset[i:(i + look_back), :]
            X.append(a)
            # Target is the next step of the specified column
            y.append(dataset[i + look_back, target_col_idx])
            
        return np.array(X), np.array(y)

    def save_scaler(self, filename="scaler.pkl"):
        """Save scaler to disk for inference."""
        os.makedirs(self.scalers_dir, exist_ok=True)
        path = os.path.join(self.scalers_dir, filename)
        joblib.dump(self.scaler, path)

    def load_scaler(self, filename="scaler.pkl"):
        path = os.path.join(self.scalers_dir, filename)
        if os.path.exists(path):
            self.scaler = joblib.load(path)
        else:
            print(f"Scaler not found at {path}, using unfitted scaler.")

    def inverse_transform(self, data, target_col_idx=0):
        """
        Inverse transform data.
        If data is (N, 1) (predictions) and scaler is (N, features),
        we need to pad the other features with zeros (or dummy values)
        to use the scaler, then extract the target column.
        """
        # Data is likely (N, 1) representing the target column (e.g., tavg)
         
        # Make a dummy array of correct shape
        # self.scaler.scale_ is array of shape (features,)
        n_features = self.scaler.n_features_in_
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        if data.shape[1] == n_features:
            # If it's the full dataset, just transform
            return self.scaler.inverse_transform(data)
            
        # Create dummy
        dummy = np.zeros((data.shape[0], n_features))
        # Place prediction in the correct column
        dummy[:, target_col_idx] = data[:, 0]
        
        # Inverse transform
        unscaled = self.scaler.inverse_transform(dummy)
        
        # Return only the target column
        return unscaled[:, target_col_idx]
