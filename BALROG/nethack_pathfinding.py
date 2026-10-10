"""
NetHack Pathfinding Algorithms for NavigateTo SubProgram
========================================================
"""

import heapq
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

class PathfindingAlgorithms:
    """Collection of pathfinding algorithms for NetHack navigation"""
    
    @staticmethod
    def parse_map(observation: str) -> np.ndarray:
        """Convert NetHack ASCII map to 2D grid"""
        lines = observation.strip().split('\n')
        max_width = max(len(line) for line in lines)
        
        # Create grid (0 = passable, 1 = wall, 2 = door, 3 = target)
        grid = np.zeros((len(lines), max_width), dtype=int)
        
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char in ['#', '-', '|']:  # Walls
                    grid[i, j] = 1
                elif char == '+':  # Closed door
                    grid[i, j] = 2
                elif char == '>':  # Stairs down (target)
                    grid[i, j] = 3
                # '.' and ' ' are passable (0)
                    
        return grid
    
    @staticmethod
    def find_player_pos(observation: str) -> Tuple[int, int]:
        """Find player position (usually marked with @)"""
        lines = observation.strip().split('\n')
        for i, line in enumerate(lines):
            if '@' in line:
                return (i, line.index('@'))
        return (0, 0)  # Default
    
    @staticmethod
    def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm
        - Optimal for single goal
        - Uses heuristic (Manhattan distance)
        """
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Priority queue: (f_score, g_score, position, path)
        frontier = [(0, 0, start, [start])]
        visited = {start}
        
        while frontier:
            f_score, g_score, current, path = heapq.heappop(frontier)
            
            if current == goal:
                return path
            
            # Check 8 directions (including diagonals)
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                next_pos = (current[0] + di, current[1] + dj)
                
                # Check bounds
                if (0 <= next_pos[0] < grid.shape[0] and 
                    0 <= next_pos[1] < grid.shape[1] and
                    next_pos not in visited):
                    
                    # Check if passable (0) or target (3) or door (2)
                    if grid[next_pos] in [0, 2, 3]:
                        new_g_score = g_score + (1.414 if di != 0 and dj != 0 else 1.0)
                        new_f_score = new_g_score + heuristic(next_pos, goal)
                        
                        heapq.heappush(frontier, (new_f_score, new_g_score, next_pos, path + [next_pos]))
                        visited.add(next_pos)
        
        return []  # No path found
    
    @staticmethod
    def theta_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Theta* pathfinding algorithm
        - Any-angle pathfinding
        - Produces smoother, more natural paths
        - Can move at any angle if line-of-sight exists
        """
        def line_of_sight(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
            """Check if there's a clear line of sight between two points"""
            x0, y0 = p1
            x1, y1 = p2
            
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x = x0
            y = y0
            
            x_inc = 1 if x1 > x0 else -1
            y_inc = 1 if y1 > y0 else -1
            
            if dx > dy:
                error = dx / 2.0
                while x != x1:
                    if grid[y, x] == 1:  # Wall
                        return False
                    error -= dy
                    if error < 0:
                        y += y_inc
                        error += dx
                    x += x_inc
            else:
                error = dy / 2.0
                while y != y1:
                    if grid[y, x] == 1:  # Wall
                        return False
                    error -= dx
                    if error < 0:
                        x += x_inc
                        error += dy
                    y += y_inc
                    
            return grid[y1, x1] != 1
        
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Similar to A* but with line-of-sight checks
        g_score = {start: 0}
        parent = {start: start}
        
        frontier = [(distance(start, goal), start)]
        visited = set()
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                return path[::-1]
            
            # Check neighbors
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + di, current[1] + dj)
                
                if (0 <= neighbor[0] < grid.shape[0] and 
                    0 <= neighbor[1] < grid.shape[1] and
                    neighbor not in visited and
                    grid[neighbor] != 1):
                    
                    # Key innovation of Theta*: check parent's line of sight
                    if current != start and line_of_sight(parent[current], neighbor):
                        # Path 2: Direct path from parent
                        new_g = g_score[parent[current]] + distance(parent[current], neighbor)
                        if neighbor not in g_score or new_g < g_score[neighbor]:
                            g_score[neighbor] = new_g
                            parent[neighbor] = parent[current]  # Skip current node
                            f_score = new_g + distance(neighbor, goal)
                            heapq.heappush(frontier, (f_score, neighbor))
                    else:
                        # Path 1: Normal A* path
                        new_g = g_score[current] + distance(current, neighbor)
                        if neighbor not in g_score or new_g < g_score[neighbor]:
                            g_score[neighbor] = new_g
                            parent[neighbor] = current
                            f_score = new_g + distance(neighbor, goal)
                            heapq.heappush(frontier, (f_score, neighbor))
        
        return []
    
    @staticmethod
    def jump_point_search(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Jump Point Search (JPS)
        - Optimal like A* but faster
        - Prunes symmetric paths
        - Great for uniform-cost grids
        """
        def jump(current: Tuple[int, int], direction: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            """Jump in direction until we find a jump point"""
            next_pos = (current[0] + direction[0], current[1] + direction[1])
            
            # Out of bounds or obstacle
            if (not (0 <= next_pos[0] < grid.shape[0] and 0 <= next_pos[1] < grid.shape[1]) or
                grid[next_pos] == 1):
                return None
                
            # Reached goal
            if next_pos == goal:
                return next_pos
                
            # Diagonal movement
            if direction[0] != 0 and direction[1] != 0:
                # Check forced neighbors
                if ((grid[current[0], next_pos[1]] == 1 and grid[current[0], next_pos[1] + direction[1]] != 1) or
                    (grid[next_pos[0], current[1]] == 1 and grid[next_pos[0] + direction[0], current[1]] != 1)):
                    return next_pos
                    
                # Recurse horizontally and vertically
                if jump(next_pos, (direction[0], 0), goal) or jump(next_pos, (0, direction[1]), goal):
                    return next_pos
            else:
                # Horizontal/vertical movement
                if direction[0] != 0:  # Vertical
                    if ((grid[next_pos[0], next_pos[1] + 1] == 1 and grid[next_pos[0] + direction[0], next_pos[1] + 1] != 1) or
                        (grid[next_pos[0], next_pos[1] - 1] == 1 and grid[next_pos[0] + direction[0], next_pos[1] - 1] != 1)):
                        return next_pos
                else:  # Horizontal
                    if ((grid[next_pos[0] + 1, next_pos[1]] == 1 and grid[next_pos[0] + 1, next_pos[1] + direction[1]] != 1) or
                        (grid[next_pos[0] - 1, next_pos[1]] == 1 and grid[next_pos[0] - 1, next_pos[1] + direction[1]] != 1)):
                        return next_pos
            
            # Continue jumping
            return jump(next_pos, direction, goal)
        
        # Use A* with jump points
        frontier = [(0, 0, start, [start])]
        visited = {start}
        
        while frontier:
            f_score, g_score, current, path = heapq.heappop(frontier)
            
            if current == goal:
                return path
                
            # Get successors using jump points
            for direction in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                jump_point = jump(current, direction, goal)
                if jump_point and jump_point not in visited:
                    new_g = g_score + np.sqrt((jump_point[0] - current[0])**2 + (jump_point[1] - current[1])**2)
                    new_f = new_g + abs(jump_point[0] - goal[0]) + abs(jump_point[1] - goal[1])
                    heapq.heappush(frontier, (new_f, new_g, jump_point, path + [jump_point]))
                    visited.add(jump_point)
                    
        return []
    
    @staticmethod
    def hierarchical_pathfinding(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Hierarchical Pathfinding (like HPA*)
        - Divides map into clusters
        - Finds path between clusters first
        - Then refines within clusters
        """
        cluster_size = 8
        
        # Create cluster graph
        height, width = grid.shape
        num_clusters_y = (height + cluster_size - 1) // cluster_size
        num_clusters_x = (width + cluster_size - 1) // cluster_size
        
        def get_cluster(pos):
            return (pos[0] // cluster_size, pos[1] // cluster_size)
        
        def get_cluster_center(cluster):
            return (cluster[0] * cluster_size + cluster_size // 2,
                    cluster[1] * cluster_size + cluster_size // 2)
        
        # First, find path between clusters
        start_cluster = get_cluster(start)
        goal_cluster = get_cluster(goal)
        
        if start_cluster == goal_cluster:
            # Same cluster, use regular A*
            return PathfindingAlgorithms.a_star(grid, start, goal)
        
        # Find entrance/exit points between clusters
        cluster_path = PathfindingAlgorithms.a_star(
            np.ones((num_clusters_y, num_clusters_x)) * 0,  # All clusters accessible
            start_cluster,
            goal_cluster
        )
        
        if not cluster_path:
            return []
        
        # Refine path through clusters
        full_path = [start]
        current_pos = start
        
        for i in range(1, len(cluster_path)):
            # Find path to exit of current cluster
            next_cluster_center = get_cluster_center(cluster_path[i])
            segment = PathfindingAlgorithms.a_star(grid, current_pos, next_cluster_center)
            if segment:
                full_path.extend(segment[1:])
                current_pos = segment[-1]
        
        # Final segment to goal
        final_segment = PathfindingAlgorithms.a_star(grid, current_pos, goal)
        if final_segment:
            full_path.extend(final_segment[1:])
            
        return full_path

def path_to_actions(path: List[Tuple[int, int]]) -> List[str]:
    """Convert path coordinates to NetHack actions"""
    if len(path) < 2:
        return []
        
    actions = []
    direction_map = {
        (-1, 0): "north",
        (1, 0): "south", 
        (0, -1): "west",
        (0, 1): "east",
        (-1, -1): "northwest",
        (-1, 1): "northeast",
        (1, -1): "southwest",
        (1, 1): "southeast"
    }
    
    for i in range(len(path) - 1):
        dy = path[i+1][0] - path[i][0]
        dx = path[i+1][1] - path[i][1]
        
        # Normalize for diagonal moves
        if dy != 0:
            dy = dy // abs(dy)
        if dx != 0:
            dx = dx // abs(dx)
            
        direction = (dy, dx)
        if direction in direction_map:
            actions.append(direction_map[direction])
            
    return actions

# Example usage
if __name__ == "__main__":
    # Example NetHack map
    example_map = """
###################
#.................#
#.####...####.....#
#.#  #...#  #.....#
#.#  +...+  #.....#
#.####...####.....#
#.......@.........#
#.................#
#.................#
#........>........#
###################
"""
    
    pf = PathfindingAlgorithms()
    grid = pf.parse_map(example_map)
    start = pf.find_player_pos(example_map)
    goal = np.argwhere(grid == 3)[0]  # Find stairs
    
    print("Map:")
    print(example_map)
    print(f"\nStart: {start}, Goal: {goal}")
    
    # Test different algorithms
    algorithms = [
        ("A*", pf.a_star),
        ("Theta*", pf.theta_star),
        ("Jump Point Search", pf.jump_point_search),
        ("Hierarchical", pf.hierarchical_pathfinding)
    ]
    
    for name, algo in algorithms:
        path = algo(grid, start, tuple(goal))
        if path:
            actions = path_to_actions(path)
            print(f"\n{name}: {len(path)} steps")
            print(f"Actions: {' -> '.join(actions[:10])}...")
        else:
            print(f"\n{name}: No path found")