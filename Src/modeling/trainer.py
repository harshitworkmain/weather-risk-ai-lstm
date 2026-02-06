import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Src.modeling.lstm_model import build_lstm_model

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.models_dir = self.config['data']['paths'].get('models', 'Models/checkpoints')
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LSTM model.
        """
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)
        
        epochs = self.config['model']['lstm']['epochs']
        batch_size = self.config['model']['lstm']['batch_size']
        
        os.makedirs(self.models_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.models_dir, "best_model.keras")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.1 if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
