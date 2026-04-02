import os
import pandas as pd
from datetime import datetime
from meteostat import Point, Daily
from Src.utils.config_loader import load_config

class DataLoader:
    def __init__(self, config: dict = None):
        self.config = config if config else load_config()

    def fetch_weather_data(self, 
                           lat: float = None, 
                           lon: float = None, 
                           start: datetime = None, 
                           end: datetime = None) -> pd.DataFrame:
        """
        Fetch historical weather data using Meteostat.
        """
        # Defaults from config if not provided
        lat = lat if lat is not None else self.config['data']['location']['latitude']
        lon = lon if lon is not None else self.config['data']['location']['longitude']
        
        if start is None:
            start_str = self.config['data']['time_range']['start_date']
            start = datetime.strptime(start_str, "%Y-%m-%d")
        
        if end is None:
            # Default to now if not specified in config or arg
            end = datetime.now()

        location = Point(lat, lon)
        data = Daily(location, start, end)
        data = data.fetch()

        if data.empty:
            raise ValueError(f"No weather data found for location ({lat}, {lon})")

        return data

    def load_csv_data(self, filepath: str) -> pd.DataFrame:
        """
        Load weather data from a CSV file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return pd.read_csv(filepath, parse_dates=True, index_col=0)

    def save_data(self, df: pd.DataFrame, filename: str = "weather_data.csv"):
        """
        Save dataframe to the processed data directory.
        """
        out_dir = self.config['data']['paths']['processed']
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        df.to_csv(path)
        print(f"Data saved to {path}")
