# Weather Risk AI System

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

## 📌 Project Overview
**Weather Risk AI** is an advanced, modular Python system designed to predict weather patterns and simulate urban disaster risks. Transitioning from a research-based Jupyter Notebook, this production-ready application leverages **Multivariate LSTM (Long Short-Term Memory)** neural networks for forecasting and **Graph Theory** for modeling risk propagation across city infrastructures.

The system is specifically tailored for scenarios like **Chennai's flood risk management**, utilizing Directed Graphs to model water flow and wind-driven risk diffusion.

## 🚀 Key Features
- **Multivariate Weather Forecasting**: Uses LSTM models to predict Temperature (`tavg`), Precipitation, Wind Speed, and Pressure based on historical data.
- **Dynamic Risk Propagation**: Simulates how risk (e.g., flooding, storm intensity) spreads across city zones using a **Directed Graph (DiGraph)**, accounting for physical flow directions.
- **Storm Path Tracking**: Implements **Dijkstra's Algorithm** to predict likely storm trajectories based on pressure gradients.
- **Flood Capacity Modeling**: Uses NetworkX **Max Flow** algorithms to estimate drainage capacity and identify bottlenecks.
- **Meteostat Integration**: Automated ingestion of historical weather data for any coordinate.

## 🏗️ System Architecture
The project follows a clean, modular `src/` layout:

```text
weather-risk-ai/
├── Config/            # Configuration management (YAML)
├── Data/              # Raw, Processed, and Output data storage
├── Docs/              # Detailed module documentation
├── Models/            # Saved LSTM checkpoints and Scalers
├── Notebooks/         # Research sandboxes
├── Src/               # Core Application Logic
│   ├── ingestion/     # Data fetching (API/CSV)
│   ├── processing/    # Cleaning, Scaling, Windowing
│   ├── modeling/      # LSTM Architecture & Training Loop
│   ├── analysis/      # Graph Risk, Flood & Storm Algorithms
│   └── utils/         # Helpers
├── Tests/             # Unit and Integration Tests
└── main.py            # CLI Entry Point
```

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/harshitworkmain/weather-risk-ai.git
   cd weather-risk-ai
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Configuration**
   - Check `Config/config.yaml` to adjust city coordinates or model hyperparameters.

## 💻 Usage

The system is driven by a unified CLI tool `main.py`.

### 1. Run Full Demo
Executes the entire pipeline: Data Fetch -> Process -> Train -> Predict -> Analyze.
```bash
python main.py demo
```

### 2. Individual Commands
**Train the Model**:
```bash
python main.py train
```

**Make a Prediction**:
```bash
python main.py predict
```

**Run Risk Analysis**:
```bash
python main.py analyze
```

## 👨‍💻 Author
**Harshit Singh**  
- **GitHub**: [harshitworkmain](https://github.com/harshitworkmain)  
- **Email**: harshit.workmain@gmail.com  

## 📄 License
This project is licensed under the MIT License.
