import heapq

class StormTracker:
    def __init__(self, grid_size=(10, 10)):
        self.grid_size = grid_size
        
        # 'Pressure' map - storms move towards low pressure.
        # Initialize with random high pressure
        self.pressure_grid = [[1013.0 for _ in range(grid_size[1])] for _ in range(grid_size[0])]

    def set_pressure(self, x, y, pressure):
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            self.pressure_grid[x][y] = pressure

    def track_storm(self, start_pos, end_pos):
        """
        Find the likely path of the storm using Dijkstra's algorithm.
        Cost = Pressure Gradient (moving to lower pressure is 'cheaper' / more likely).
        Actually, physically, it flows TO low pressure. 
        So neighbors with LOWER pressure should have LOWER cost to traverse.
        Cost(u -> v) = Pressure(v)
        """
        start = start_pos
        target = end_pos # Or just find path to global minimum? 
        # For this function, let's pathfind between two points as a demo.
        
        rows, cols = self.grid_size
        distances = { (r, c): float('inf') for r in range(rows) for c in range(cols) }
        distances[start] = 0
        pq = [(0, start)]
        previous = {}
        
        directions = [(-1,0), (1,0), (0,-1), (0,1)] # 4-connectivity
        
        while pq:
            current_dist, (r, c) = heapq.heappop(pq)
            
            if (r, c) == target:
                break
            
            if current_dist > distances[(r, c)]:
                continue
                
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Cost is the pressure at the destination node
                    # Lower pressure = easier to move there?
                    # Let's normalize: Cost = Pressure(nr, nc)
                    weight = self.pressure_grid[nr][nc]
                    distance = current_dist + weight
                    
                    if distance < distances[(nr, nc)]:
                        distances[(nr, nc)] = distance
                        previous[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (distance, (nr, nc)))
        
        # Reconstruct path
        path = []
        curr = target
        if distances[curr] == float('inf'):
            return [] # No path
            
        while curr in previous:
            path.append(curr)
            curr = previous[curr]
        path.append(start)
        return path[::-1]
