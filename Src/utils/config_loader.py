import yaml
import os
from pathlib import Path

def load_config(config_path: str = "Config/config.yaml") -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Relative path to the config file.
        
    Returns:
        dict: The configuration dictionary.
    """
    # Get the project root directory (assuming this script is in Src/utils)
    # If run from project root, Path(config_path) works. 
    # If run from elsewhere, we might need absolute resolution.
    # For now, we assume execution from project root.
    
    if not os.path.exists(config_path):
        # Try to find it relative to this file if it fails
        base_path = Path(__file__).parent.parent.parent
        config_path = base_path / config_path
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config
