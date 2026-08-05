# Weather Risk AI System – Data Engineer Resume Guide

This document provides a **100% authentic, defendable, and metric-backed description** of the `weather-risk-ai-lstm` project tailored specifically for **Data Engineer, Analytics Engineer, and ETL/Pipeline Engineer** roles at top Indian technology companies (e.g., Swiggy, Zomato, Flipkart, Razorpay, CRED, PhonePe, Meesho, InMobi) and global tech giants (e.g., Google, Amazon, Microsoft, Uber, Snowflake, Databricks).

---

## 1. Quick Copy-Paste Resume Bullet Points (Google X-Y-Z Formula)

> **Formula Structure:** *Accomplished [X] as measured by [Y], by doing [Z]*

### Variant A: Comprehensive Data Engineer Format
* **Built an automated multi-source weather ETL data pipeline**, extracting and ingesting **1,460+ daily observations (2020–2024)** across **6 meteorological features** with **0% data loss**, by integrating Meteostat REST APIs, point coordinate lookups, and dynamic date boundary parameters.
* **Engineered a data cleaning & quality transformation module**, ensuring **100% time-series data completeness**, by designing sequential forward-fill (`ffill()`) and backward-fill (`bfill()`) missing value imputation routines in Pandas to preserve temporal causality.
* **Constructed a 3D feature tensor transformation engine**, converting tabular weather data into **sliding window arrays of shape `(N, 10, 6)`**, by deploying Scikit-Learn `MinMaxScaler` normalization, zero-leakage training fits, and binary artifact serialization (`joblib`).
* **Designed a 3-tier data lake directory architecture** (`Data/raw`, `Data/processed`, `Data/outputs`), isolating ingestion, transformation, and output artifacts, by implementing centralized YAML configuration management (`config.yaml`) and dynamic path loader utilities.
* **Orchestrated the end-to-end data pipeline lifecycle**, reducing manual pipeline run time to **a single CLI command (`python main.py demo`)**, by engineering modular Python handlers (`ingestion`, `processing`, `modeling`, `analysis`, `utils`) with `Typer` CLI and `PyTest` validation.

---

### Variant B: Concise Format (3 Bullet Points for Data Engineer Resumes)
* **Architected an automated time-series ETL pipeline** using Python, Pandas, and Meteostat API, ingesting 4 years of daily meteorological data (1,460+ records) across 6 features (`tavg`, `tmin`, `tmax`, `prcp`, `wspd`, `pres`).
* **Developed data quality & transformation modules**, handling missing values via temporal `ffill`/`bfill` imputation, normalizing features via `MinMaxScaler`, serializing transformation objects (`scaler.pkl`), and generating 10-day sliding window feature tensors `(N, 10, 6)`.
* **Structured a modular 5-component data pipeline** with YAML configuration management, isolated data lake directory tiers (`Data/raw`, `Data/processed`, `Data/outputs`), `Typer` CLI command automation, and `PyTest` pipeline verification.

---

## 2. 30-Second Interview Elevator Pitch (Data Engineer Persona)

> *"In this project, I engineered an automated time-series ETL data pipeline for meteorological and spatial analytics. I built an ingestion module that extracts 4 years of daily weather observations for Chennai via the Meteostat API, standardizing 6 core features into structured Pandas dataframes. To ensure data quality, I implemented temporal forward and backward fill imputation routines that eliminate missing values without causing temporal data leakage. I then transformed these 2D dataframes into 3D sliding-window feature tensors `(N, 10, 6)` for downstream ML consumers, persisting scaler binaries with `joblib`. The pipeline is designed around a multi-tier storage layout (`Data/raw`, `Data/processed`, `Data/outputs`) driven by a central YAML configuration file and automated via a Typer CLI."*

---

## 3. Deep-Dive Technical Data Engineering Questions

### Q1: "Walk me through your ETL pipeline architecture from data ingestion to storage."
> **Answer:** *"The data pipeline (`Src/ingestion` and `Src/processing`) follows a clean 3-stage ETL design:
> 1. **Extract (`data_loader.py`):** Uses the `Meteostat` API to fetch daily weather data based on geographic coordinates (Latitude: 13.0827, Longitude: 80.2707) and date boundaries defined in `Config/config.yaml`. Supports fallback to CSV loading (`load_csv_data`).
> 2. **Transform (`preprocessor.py` & `feature_engine.py`):** 
>    - `clean_data()` sorts by date index and applies `ffill().bfill().fillna(0)` to guarantee zero NaNs.
>    - `scale_data()` applies `MinMaxScaler(0, 1)` and serializes the fit scaler binary (`scaler.pkl`) via `joblib`.
>    - `create_sliding_window()` converts the 2D matrix into 3D tensors `(Samples, 10, 6)` for 10-day lookback windowing.
> 3. **Load / Storage:** Persists processed dataframes to `Data/processed/weather_data.csv` and serialized assets to `Data/processed/scaler.pkl`."*

---

### Q2: "How did you prevent data leakage during feature scaling and data transformations?"
> **Answer:** *"In time-series data engineering, scaling or imputing based on global statistics introduces target leakage from future observations. In `Src/processing/feature_engine.py`:
> - The `scale_data` method accepts an `is_training` boolean flag.
> - During training (`is_training=True`), `scaler.fit_transform()` computes $X_{min}$ and $X_{max}$ strictly on training rows and serializes the scaler artifact to `Data/processed/scaler.pkl` using `joblib`.
> - During inference (`is_training=False`), the pipeline loads the pre-fit scaler binary and executes `.transform()` strictly, enforcing target separation.
> - For missing data, `preprocessor.py` uses chronological forward-fill (`ffill()`) before backward-fill (`bfill()`) to preserve temporal ordering."*

---

### Q3: "How is your storage tiering and directory layout structured?"
> **Answer:** *"To support data governance and lineage, the repository maintains explicit directory isolation controlled via `Config/config.yaml`:
> - `Data/raw/`: Landing directory for unmodified external API payloads or raw CSV dumps.
> - `Data/processed/`: Staging directory for cleaned dataframes (`weather_data.csv`) and serialized transformation binaries (`scaler.pkl`).
> - `Data/outputs/`: Final destination for downstream analytical outputs, graph risk tables, and model evaluation metrics.
> 
> All file paths are dynamically loaded via `config_loader.py`, eliminating hardcoded strings across modules."*

---

### Q4: "How do you handle schema variations or missing values in raw API responses?"
> **Answer:** *"The raw Meteostat dataset can contain missing values due to sensor downtime. In `preprocessor.py`, data is first sorted by date index to ensure chronological ordering. We then apply `df.ffill().bfill()`. Forward-fill propagates the last known valid observation forward, maintaining physical continuity. Backward-fill resolves any initial missing values at the dataset boundary. A final `.fillna(0)` guard clause ensures that no `NaN` values ever reach downstream transformation tensors."*

---

## 4. Data Engineer Codebase Technical Reference Matrix

| Pipeline Component | Data Engineering Technical Implementation | Code File Path |
| :--- | :--- | :--- |
| **API Ingestion** | Meteostat `Daily(location, start, end).fetch()` | `Src/ingestion/data_loader.py` |
| **Coordinate Lookup** | Point(13.0827, 80.2707, elev=6m) | `Config/config.yaml` & `data_loader.py` |
| **Time Window Span** | 4 Years (Jan 1, 2020 – Jan 1, 2024) $\approx$ 1,461 daily records | `Config/config.yaml` |
| **Schema Features** | 6 Features: `tavg`, `tmin`, `tmax`, `prcp`, `wspd`, `pres` | `main.py` & `feature_engine.py` |
| **Imputation Strategy** | Sequential `ffill().bfill().fillna(0)` | `Src/processing/preprocessor.py` |
| **Data Scaling** | `MinMaxScaler(feature_range=(0,1))` | `Src/processing/feature_engine.py` |
| **Binary Serialization** | `joblib.dump(self.scaler, 'Data/processed/scaler.pkl')` | `Src/processing/feature_engine.py` |
| **Tensor Windowing** | 3D Array Generator `(Samples, 10, 6)` | `Src/processing/feature_engine.py` |
| **Storage Tiers** | `Data/raw`, `Data/processed`, `Data/outputs` | `Config/config.yaml` |
| **Config Loader** | PyYAML loader returning nested dictionary | `Src/utils/config_loader.py` |
| **CLI Automation** | Python `typer` CLI (`train`, `predict`, `analyze`, `demo`) | `main.py` |

---

## 5. Recommended Skills for Resume Sidebar (Data Engineer Focus)

* **Data Engineering & ETL:** Data Pipeline Design, Data Ingestion APIs, Data Quality & Imputation, Time-Series Transformations, 3D Tensor Windowing, Data Leakage Prevention
* **Data Storage & Architecture:** Data Lake Storage Tiering (`raw`, `processed`, `outputs`), Binary Artifact Serialization (`joblib`), Schema Validation, PyYAML Configuration Management
* **Languages & Libraries:** Python 3.10+, Pandas, NumPy, Scikit-Learn, Meteostat API, Typer CLI, PyTest
* **Software Engineering & DevOps:** Modular Software Design, CLI Automation, Git/GitHub Version Control, `setup.py` Packaging
