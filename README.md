# 🌪️ Weather Risk AI System

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

An advanced, production-grade AI system designated to forecast hyper-local weather anomalies using Deep Learning and simulate urban risk propagation—such as cascading floods or storm spread—via Graph Theory. 

Transitioning from local proof-of-concepts to a modular analytics pipeline, this platform is deeply tailored for major urban environments (e.g., Chennai's coastal geography), equipping city planners and emergency teams with predictive insights.

---

## 🎯 Key Capabilities & Methodology

### 1. Neural Weather Forecasting
Leverages **Multivariate Long Short-Term Memory (LSTM)** models to detect complex temporal dependencies across diverse meteorological features (Temperature, Precipitation, Wind Speed, Pressure).

> **Model Training Lifecycle**  
> *Tracking multi-epoch convergence for optimal sequence learning.*
> 
> ![Training Results](Docs/images/model_training_results.png)

### 2. Graph-Theoretical Risk Dynamics
Models the city as a comprehensive **Directed Graph (DiGraph)**, where nodes represent urban zones and edges map the topological flow of natural elements (e.g., water drainage, wind corridors).

> **Urban Risk Diffusion**  
> *Visualizing how localized phenomena (like heavy rainfall) propagate to adjacent boroughs.*
>
> ![Risk Propagation](Docs/images/risk_propagation_graph.png)

### 3. Geospatial Threat Mapping
Visualizes static and dynamic risk metrics across distinct topologies, identifying permanent high-risk zones.

> **Geospatial Hotspots**
>
> ![Risk Zones](Docs/images/geospatial_risk_zones.png)

### 4. Flood Capacity & Network Flow
Utilizes NetworkX's **Max Flow algorithms** to estimate optimal and strained drainage capacities across interconnected city basins.

> **Flood Network Topography**
>
> ![Flood Risk](Docs/images/flood_risk_network.png)

### 5. Storm Trajectory Prediction
Integrates **Dijkstra's Shortest Path Algorithm** with air pressure gradient data to predict likely pathways for incoming storm cells.

> **Predicted Path Algorithms**
>
> ![Storm Tracking](Docs/images/storm_tracking_network.png)

---

## 🏗️ System Architecture

Our repository follows strict ML DevOps standards, isolating research from production workloads.

```text
weather-risk-ai-lstm/
├── Config/            # Configuration environments (YAML)
├── Data/              # Raw ingested APIs, features, artifacts
├── Docs/              # Module documentation and media assets
│   └── images/        # ML visual evaluations
├── Models/            # Persisted .keras checkpoints, robust scalers
├── Notebooks/         # EDA and sandbox research 
├── Src/               # Core Execution Logic
│   ├── ingestion/     # Automations for Meteostat & API fetch
│   ├── processing/    # Scaling, Imputation, Sliding Windows
│   ├── modeling/      # Keras/TF Architectures
│   ├── analysis/      # Graph & NetworkX topology algorithms
│   └── utils/         # Telemetry & Config loader
├── Tests/             # PyTest Integration frameworks
└── main.py            # Master CLI Entry Point
```

---

## 🚀 Installation

Ensure you have a modern Python 3.10+ environment set up.

1. **Clone the Source**
   ```bash
   git clone https://github.com/harshitworkmain/weather-risk-ai-lstm.git
   cd weather-risk-ai-lstm
   ```

2. **Install Core Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Environment**
   Inspect `Config/config.yaml` to adjust city parameters, API endpoints, or model hyper-parameters (such as sequence look-back arrays or LSTM layer width).

---

## 💻 CLI Usage Patterns

The pipeline is unified under a single robust CLI tool built tightly with Python `typer`.

### Complete Integration Demo
Executes the full local workflow seamlessly: `Ingest -> Preprocess -> Train -> Predict -> Analyze`.
```bash
python main.py demo
```

### Granular Execution
**Re-Train the LSTM Model**:
```bash
python main.py train
```

**Run Autoregressive Predictions**:
```bash
python main.py predict
```

**Execute Urban Graph Analysis**:
```bash
python main.py analyze
```

---

## 👨‍💻 Maintainer

**Harshit Singh**  
- **GitHub**: [@harshitworkmain](https://github.com/harshitworkmain)  
- **Contact**: harshit.workmain@gmail.com  

## 📄 License
This project is licensed under the MIT License.
