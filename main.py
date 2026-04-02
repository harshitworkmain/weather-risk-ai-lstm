import typer
from datetime import datetime
from Src.ingestion.data_loader import DataLoader
from Src.processing.preprocessor import clean_data
from Src.processing.feature_engine import FeatureEngine
from Src.modeling.trainer import Trainer
from Src.analysis.graph_network import CityGraph
from Src.utils.config_loader import load_config

app = typer.Typer()

@app.command()
def train():
    """Run the training pipeline."""
    print("Loading config...")
    config = load_config()
    
    print("Fetching data...")
    loader = DataLoader(config)
    # Using defaults from config
    df = loader.fetch_weather_data() 
    
    print("Cleaning data...")
    df_clean = clean_data(df)
    
    print("Feature Engineering...")
    engine = FeatureEngine(config)
    # Select columns - in real app, specify in config
    features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
    df_features = df_clean[features]
    
    df_scaled = engine.scale_data(df_features, is_training=True)
    
    X, y = engine.create_sliding_window(df_scaled, target_col_idx=0)
    # Reshape X for LSTM: (Samples, Timesteps, Features)
    
    print(f"Training data shape: {X.shape}")
    
    print("Training Model...")
    trainer = Trainer(config)
    trainer.train(X, y)
    print("Training Complete. Model saved.")

@app.command()
def predict(days: int = 7):
    """Predict future weather/risk."""
    config = load_config()
    loader = DataLoader(config)
    df = loader.fetch_weather_data()
    
    # Preprocess
    df_clean = clean_data(df)
    features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
    df_features = df_clean[features]
    
    engine = FeatureEngine(config)
    # Load existing scaler if possible, else fit new (for demo we fit new as we don't persist well yet)
    # ideally we load
    engine.scale_data(df_features, is_training=True) 
    
    # Get last sequence
    look_back = config['model']['lstm']['look_back']
    last_sequence_data = df_features.values[-look_back:]
    
    # Scale
    last_sequence_scaled = engine.scaler.transform(last_sequence_data)
    
    # Load model
    from tensorflow.keras.models import load_model
    import numpy as np
    model_path = config['data']['paths'].get('models', 'Models/checkpoints') + "/best_model.keras"
    try:
        model = load_model(model_path)
    except:
        print("Model not found. Run 'train' first.")
        return

    # Predict
    # This is a simple recursive prediction (naive) assuming we only need prior data, 
    # but strictly if we have multivariate input, we need future input for other features usually!
    # A pure autoregressive multivariate forecast needs to forecast ALL features to feed back in.
    # For now, we will just predict the NEXT step 'tavg' given current history.
    
    current_seq = last_sequence_scaled.reshape(1, look_back, len(features))
    
    prediction_scaled = model.predict(current_seq)
    
    # Inverse transform
    prediction = engine.inverse_transform(prediction_scaled, target_col_idx=0)
    
    print(f"Predicted TAVG for next day: {prediction[0][0]:.2f}°C")

@app.command()
def analyze():
    """Run graph risk analysis."""
    print("Building City Graph...")
    cg = CityGraph()
    cg.build_chennai_graph()
    
    print("Simulating Risk Propagation...")
    risks = cg.propagate_risk()
    
    print("Final Risk Scores:")
    for city, risk in risks.items():
        print(f"{city}: {risk:.4f}")

@app.command()
def demo():
    """Run full demo."""
    train()
    predict()
    analyze()

if __name__ == "__main__":
    app()
