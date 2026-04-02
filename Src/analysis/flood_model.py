import networkx as nx

def calculate_max_flow(graph: nx.DiGraph, source: str, sink: str):
    """
    Calculate maximum water flow capacity between two points.
    
    Args:
        graph (nx.DiGraph): The city graph with 'capacity' attributes on edges.
                            If 'capacity' is missing, it assumes infinite or weight-based.
        source (str): Starting node.
        sink (str): Ending node.
        
    Returns:
        tuple: (flow_value, flow_dict)
    """
    if source not in graph or sink not in graph:
        # In a real app, maybe find nearest node. 
        # Here we just return 0 if nodes don't exist.
        return 0, {}
    
    # Ensure edges have capacity. If not, map weight to capacity?
    # Or set default.
    for u, v, data in graph.edges(data=True):
        if 'capacity' not in data:
            # Heuristic: inverse of weight (distance) -> closer means higher capacity?
            # Or just constant. Let's say constant 10 for demo.
            data['capacity'] = 10 

    try:
        flow_value, flow_dict = nx.maximum_flow(graph, source, sink)
        return flow_value, flow_dict
    except Exception as e:
        print(f"Error calculating max flow: {e}")
        return 0, {}
