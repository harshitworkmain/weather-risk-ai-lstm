import pytest
import networkx as nx
from Src.analysis.graph_network import CityGraph

def test_risk_propagation():
    cg = CityGraph()
    G = cg.G
    
    # Manual setup
    G.add_node("A", risk_score=1.0)
    G.add_node("B", risk_score=0.0)
    G.add_node("C", risk_score=0.0)
    
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    
    # Propagate
    risks = cg.propagate_risk(iterations=1, decay_factor=0.5)
    
    # A should stay 1.0 (max logic)
    assert risks["A"] >= 1.0
    
    # B neighbor is A (risk 1.0). Algo: sum(neighbors) / len. 
    # B has neighbor A (undirected? code assumed add_edge u,v and v,u).
    # If B connected to A and C. Neighbors = [A, C]. 
    # Incoming = Risk(A) + Risk(C) = 1 + 0 = 1.
    # Avg = 1 / 2 = 0.5. * decay(0.5) = 0.25.
    # New B = max(0, 0.25) = 0.25.
    
    # Note: Logic in code was: `avg_risk = (incoming_risk / len(neighbors)) * decay_factor`
    # Let's verify B's risk is > 0
    assert risks["B"] > 0
