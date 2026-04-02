from setuptools import setup, find_packages

setup(
    name="weather_risk_ai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "networkx",
        "shapely",
        "tensorflow",
        "scikit-learn",
        "meteostat",
        "geopandas",
        "pyyaml",
        "python-dotenv",
        "typer"
    ],
)
