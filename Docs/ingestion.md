# Ingestion Module Documentation

## Overview
The `Src/ingestion` module is responsible for retrieving raw weather data from external sources. It abstracts the complexity of API calls and file handling, providing a standard DataFrame interface for the rest of the application.

## Components

### `data_loader.py`

#### `DataLoader` Class
The main entry point for data ingestion.

- **Methods**:
  - `fetch_weather_data(lat, lon, start, end)`: 
    - Connects to the **Meteostat** API.
    - Retrieives daily historical weather data for the specified location and date range.
    - Defaults to the values specified in `Config/config.yaml`.
  - `load_csv_data(filepath)`:
    - Fallback method to load a local CSV file.
    - Useful for offline training or using custom datasets.
  - `save_data(df, filename)`:
    - Saves the fetched/processed DataFrame to the `Data/processed/` directory for caching.

## Usage Example
```python
from Src.ingestion.data_loader import DataLoader
from Src.utils.config_loader import load_config

config = load_config()
loader = DataLoader(config)

# Fetch data for Chennai
df = loader.fetch_weather_data()
print(df.head())
```
