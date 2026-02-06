import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the weather data:
    1. Handle missing values (impute with mean).
    2. Remove outliers (optional - sticking to basic cleaning for now).
    """
    # Ensure it's sorted by date
    df = df.sort_index()

    # Fill missing values with the mean of the column
    # Forward fill first for time-series continuity, then mean for remaining
    df = df.ffill().bfill()
    
    # If any remain (shouldn't), fill with 0
    df = df.fillna(0)
    
    return df
