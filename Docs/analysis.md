# Analysis Module Documentation

## Overview
The `Src/analysis` module contains the algorithmic core for risk assessment. It moves beyond simple forecasting to simulate physical interactions in the city using Graph Theory.

## Components

### `graph_network.py`
Models the city as a network of connected zones.

#### `CityGraph` Class
- **Structure**: Uses a `networkx.DiGraph` (Directed Graph) to model asymmetric flows (e.g., water flowing downhill or wind blowing East to West).
- **Methods**:
  - `build_chennai_graph()`: Constructs the nodes (localities like T. Nagar, Adyar) and edges (connections).
  - `propagate_risk(iterations, decay_factor)`:
    - Simulates the diffusion of risk over time.
    - Risk flows from **Predecessors** (upstream nodes) to Successors.
    - **Logic**: `NewRisk = max(CurrentRisk, Average(Incoming_Risks) * Decay)`.

### `storm_tracker.py` (New Feature)
Predicts storm movement.

#### `StormTracker` Class
- **Algorithm**: **Dijkstra’s Algorithm**.
- **Logic**: 
  - Treats the region as a grid.
  - A Storm seeks the path of least resistance (towards **Low Pressure**).
  - High Pressure areas act as "high cost" barriers.
  - Calculates the optimal path from current location to the lowest pressure sink.

### `flood_model.py`
Estimates drainage capacity.

- **`calculate_max_flow(graph, source, sink)`**:
  - Wraps the **Edmonds-Karp** or Preflow-push algorithm.
  - Determines the maximum volume of water that can flow through the network before flooding occurs.

## Usage Example
```python
from Src.analysis.graph_network import CityGraph

cg = CityGraph()
cg.build_chennai_graph()
final_risks = cg.propagate_risk(iterations=5)
```
