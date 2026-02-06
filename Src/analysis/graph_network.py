import networkx as nx
import numpy as np
import random

class CityGraph:
    def __init__(self):
        self.G = nx.DiGraph() # Using DiGraph as per validation report recommendation

    def build_chennai_graph(self):
        """
        Construct a static graph representing Chennai's key localities.
        In a real app, this might come from a shapefile or DB.
        """
        locations = [
            "Marina Beach", "T. Nagar", "Adyar", "Velachery", 
            "Guindy", "Tambaram", "Anna Nagar", "Mylapore"
        ]
        
        # Add nodes with initial random risk (or 0)
        for loc in locations:
            self.G.add_node(loc, risk_score=random.uniform(0.1, 0.5))

        # Add edges (representing proximity or flow) - Hypothetical connections
        # Directional: Source -> Target (Risk Flows this way)
        edges = [
            ("Marina Beach", "Mylapore", 1.0),    # Sea breeze / drainage
            ("Mylapore", "Adyar", 1.2),
            ("Adyar", "Guindy", 1.5),
            ("Guindy", "Velachery", 0.8),         # Runoff to Velachery
            ("T. Nagar", "Anna Nagar", 2.0),
            ("T. Nagar", "Guindy", 1.1),
            ("Tambaram", "Guindy", 3.0),          # Suburbs draining in
            ("Velachery", "Adyar", 1.0)           # Backflow?
        ]
        
        for u, v, w in edges:
            self.G.add_edge(u, v, weight=w)
            
        return self.G

    def propagate_risk(self, iterations: int = 5, decay_factor: float = 0.8):
        """
        Simulate risk diffusion across the city.
        High risk areas spread risk to neighbors (Successors).
        
        We iterate nodes and calculate their NEW risk based on INCOMING edges (Predecessors).
        """
        G = self.G
        
        for i in range(iterations):
            new_risks = {}
            for node in G.nodes():
                current_risk = G.nodes[node].get('risk_score', 0)
                
                # Risk comes FROM upstream nodes (Predecessors)
                predecessors = list(G.predecessors(node))
                
                if not predecessors:
                    new_risks[node] = current_risk
                    continue
                
                # Incoming risk from predecessors
                # Incorporate edge weights? For now just sum of source risks.
                incoming_risk = sum(G.nodes[n].get('risk_score', 0) for n in predecessors)
                
                # Average spread
                avg_risk = (incoming_risk / len(predecessors)) * decay_factor
                
                # Update logic: Max of current or new spread (conservative risk)
                new_risks[node] = max(current_risk, avg_risk) 
            
            # Apply updates
            for node, risk in new_risks.items():
                G.nodes[node]['risk_score'] = risk
                
        return nx.get_node_attributes(G, 'risk_score')
