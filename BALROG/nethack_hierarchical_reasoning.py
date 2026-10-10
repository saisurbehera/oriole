"""
Hierarchical Reasoning System for NetHack
=========================================

HPA* adapted for NetHack's structure:
- Rooms as clusters
- Corridors as connections
- Doors as gateways
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import networkx as nx
import numpy as np

class LocationType(Enum):
    ROOM = "room"
    CORRIDOR = "corridor"
    DOORWAY = "doorway"
    STAIRS = "stairs"
    UNKNOWN = "unknown"

@dataclass
class DungeonRegion:
    """A region in the dungeon (room, corridor segment, etc.)"""
    id: str
    type: LocationType
    bounds: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    tiles: Set[Tuple[int, int]]
    connections: Set[str]  # IDs of connected regions
    features: Dict[str, any]  # Items, monsters, etc.
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bounds[0] + self.bounds[2]) // 2,
                (self.bounds[1] + self.bounds[3]) // 2)
    
    def contains(self, pos: Tuple[int, int]) -> bool:
        return pos in self.tiles

class HierarchicalDungeonMap:
    """High-level map representation for hierarchical planning"""
    
    def __init__(self):
        self.regions: Dict[str, DungeonRegion] = {}
        self.graph = nx.Graph()  # Region connectivity graph
        self.current_region_id: Optional[str] = None
        self.explored_tiles: Set[Tuple[int, int]] = set()
        
    def update_from_observation(self, obs: Dict):
        """Update map from NetHack observation"""
        ascii_map = obs.get("text", {}).get("long_term_context", "")
        
        # Parse visible area
        new_regions = self._identify_regions(ascii_map)
        
        # Merge with existing knowledge
        for region in new_regions:
            self._merge_region(region)
            
        # Update connectivity
        self._update_connections()
        
        # Update current location
        player_pos = self._find_player_position(ascii_map)
        if player_pos:
            self.current_region_id = self._get_region_at(player_pos)
    
    def _identify_regions(self, ascii_map: str) -> List[DungeonRegion]:
        """Identify rooms and corridors from ASCII map"""
        lines = ascii_map.strip().split('\n')
        if not lines:
            return []
            
        height = len(lines)
        width = max(len(line) for line in lines)
        
        # Create grid
        grid = np.zeros((height, width), dtype=int)
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == '.':  # Floor
                    grid[i, j] = 1
                elif char == '#':  # Corridor
                    grid[i, j] = 2
                elif char == '+':  # Door
                    grid[i, j] = 3
                elif char == '>':  # Stairs down
                    grid[i, j] = 4
                elif char == '<':  # Stairs up
                    grid[i, j] = 5
                    
        # Flood fill to identify connected regions
        regions = []
        visited = np.zeros_like(grid, dtype=bool)
        region_id = 0
        
        for i in range(height):
            for j in range(width):
                if grid[i, j] > 0 and not visited[i, j]:
                    region = self._flood_fill_region(grid, visited, i, j, region_id)
                    if region:
                        regions.append(region)
                        region_id += 1
                        
        return regions
    
    def _flood_fill_region(self, grid, visited, start_y, start_x, region_id) -> Optional[DungeonRegion]:
        """Flood fill to identify a connected region"""
        stack = [(start_y, start_x)]
        tiles = set()
        region_type = grid[start_y, start_x]
        
        min_y, min_x = start_y, start_x
        max_y, max_x = start_y, start_x
        
        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
                
            visited[y, x] = True
            tiles.add((y, x))
            
            # Update bounds
            min_y, min_x = min(min_y, y), min(min_x, x)
            max_y, max_x = max(max_y, y), max(max_x, x)
            
            # Check neighbors (4-connected for rooms, 8-connected for corridors)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if region_type == 2:  # Corridor
                directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
                
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and
                    not visited[ny, nx] and grid[ny, nx] == region_type):
                    stack.append((ny, nx))
        
        if len(tiles) < 3:  # Ignore tiny regions
            return None
            
        # Determine region type
        if region_type == 1:
            loc_type = LocationType.ROOM
        elif region_type == 2:
            loc_type = LocationType.CORRIDOR
        elif region_type == 3:
            loc_type = LocationType.DOORWAY
        elif region_type in [4, 5]:
            loc_type = LocationType.STAIRS
        else:
            loc_type = LocationType.UNKNOWN
            
        return DungeonRegion(
            id=f"{loc_type.value}_{region_id}",
            type=loc_type,
            bounds=(min_y, min_x, max_y, max_x),
            tiles=tiles,
            connections=set(),
            features={}
        )
    
    def _merge_region(self, new_region: DungeonRegion):
        """Merge new region with existing knowledge"""
        # Check if it overlaps with existing regions
        for existing_id, existing in self.regions.items():
            overlap = new_region.tiles & existing.tiles
            if len(overlap) > len(new_region.tiles) * 0.5:
                # Significant overlap, merge
                existing.tiles |= new_region.tiles
                existing.bounds = (
                    min(existing.bounds[0], new_region.bounds[0]),
                    min(existing.bounds[1], new_region.bounds[1]),
                    max(existing.bounds[2], new_region.bounds[2]),
                    max(existing.bounds[3], new_region.bounds[3])
                )
                return
                
        # No significant overlap, add as new region
        self.regions[new_region.id] = new_region
        self.graph.add_node(new_region.id, region=new_region)
    
    def _update_connections(self):
        """Update connectivity between regions"""
        # Check for adjacent regions
        for id1, region1 in self.regions.items():
            for id2, region2 in self.regions.items():
                if id1 >= id2:  # Avoid duplicates
                    continue
                    
                # Check if regions are adjacent
                if self._regions_adjacent(region1, region2):
                    region1.connections.add(id2)
                    region2.connections.add(id1)
                    self.graph.add_edge(id1, id2)
    
    def _regions_adjacent(self, r1: DungeonRegion, r2: DungeonRegion) -> bool:
        """Check if two regions are adjacent"""
        # Check if any tiles are neighbors
        for y1, x1 in r1.tiles:
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                if (y1 + dy, x1 + dx) in r2.tiles:
                    return True
        return False
    
    def plan_high_level_path(self, target_feature: str) -> List[str]:
        """Plan path to region containing target feature"""
        if not self.current_region_id:
            return []
            
        # Find regions with target feature
        target_regions = []
        for region_id, region in self.regions.items():
            if target_feature in region.features:
                target_regions.append(region_id)
                
        if not target_regions:
            return []  # Target not found
            
        # Find shortest path to any target region
        shortest_path = None
        for target_id in target_regions:
            try:
                path = nx.shortest_path(self.graph, self.current_region_id, target_id)
                if shortest_path is None or len(path) < len(shortest_path):
                    shortest_path = path
            except nx.NetworkXNoPath:
                continue
                
        return shortest_path or []

class HierarchicalNavigator:
    """Navigate using hierarchical planning"""
    
    def __init__(self):
        self.dungeon_map = HierarchicalDungeonMap()
        self.current_plan: List[str] = []
        self.local_pathfinder = LocalPathfinder()
        
    def update(self, obs: Dict):
        """Update map from observation"""
        self.dungeon_map.update_from_observation(obs)
        
    def navigate_to(self, target: str) -> str:
        """Get next action to navigate to target"""
        # High-level planning
        if not self.current_plan:
            self.current_plan = self.dungeon_map.plan_high_level_path(target)
            
        if not self.current_plan:
            return "search"  # Explore to find target
            
        # Get next region to move to
        if len(self.current_plan) > 1:
            next_region_id = self.current_plan[1]
            next_region = self.dungeon_map.regions[next_region_id]
            
            # Local pathfinding to next region
            action = self.local_pathfinder.move_to_region(
                self.dungeon_map.current_region_id,
                next_region
            )
            
            # Check if we've entered the next region
            if self.dungeon_map.current_region_id == next_region_id:
                self.current_plan.pop(0)
                
            return action
        else:
            # We're in the target region, move to specific target
            return self.local_pathfinder.move_to_target(target)

class LocalPathfinder:
    """Low-level pathfinding within and between regions"""
    
    def move_to_region(self, current_region_id: str, target_region: DungeonRegion) -> str:
        """Move from current region towards target region"""
        # Find doorway or connection point
        # For now, move towards center of target region
        return self._move_towards(target_region.center)
        
    def move_to_target(self, target: str) -> str:
        """Move to specific target within current region"""
        # This would use A* or Theta* for precise movement
        return "search"  # Placeholder
        
    def _move_towards(self, target_pos: Tuple[int, int]) -> str:
        """Simple movement towards target position"""
        # In real implementation, would use current position
        # and calculate best direction
        return "north"  # Placeholder

# ============================================================================
# Hierarchical Sub-Programs for NetHack
# ============================================================================

class ExploreLevel(SubProgram):
    """High-level: Explore entire dungeon level"""
    
    def __init__(self):
        super().__init__("explore_level")
        self.navigator = HierarchicalNavigator()
        self.unexplored_regions: Set[str] = set()
        
    def execute(self, state: Dict) -> str:
        self.navigator.update(state)
        
        # Get list of all regions
        all_regions = set(self.navigator.dungeon_map.regions.keys())
        explored_regions = set()  # Regions we've fully explored
        
        # Find unexplored regions
        self.unexplored_regions = all_regions - explored_regions
        
        if self.unexplored_regions:
            # Navigate to nearest unexplored region
            target_region = min(self.unexplored_regions,
                              key=lambda r: self._distance_to_region(r))
            return self.navigator.navigate_to(f"explore_{target_region}")
        else:
            return "search"  # Look for secret areas

class ClearDungeon(SubProgram):
    """Highest level: Clear entire dungeon"""
    
    def __init__(self):
        super().__init__("clear_dungeon")
        self.sub_goals = [
            "explore_level",
            "collect_items", 
            "find_stairs",
            "descend"
        ]
        self.current_goal_index = 0
        
    def execute(self, state: Dict) -> str:
        current_goal = self.sub_goals[self.current_goal_index]
        
        if current_goal == "explore_level":
            explorer = ExploreLevel()
            action = explorer.execute(state)
            
            if state.get("level_explored", False):
                self.current_goal_index += 1
                
        elif current_goal == "collect_items":
            # Navigate to any visible items
            if state.get("items_visible"):
                return "pickup"
            else:
                self.current_goal_index += 1
                
        elif current_goal == "find_stairs":
            if state.get("stairs_found"):
                self.current_goal_index += 1
            else:
                return "search"
                
        elif current_goal == "descend":
            if state.get("on_stairs"):
                return "down"
            else:
                # Navigate to stairs
                navigator = HierarchicalNavigator()
                return navigator.navigate_to("stairs")
                
        return "wait"

# ============================================================================
# Example: How it works in practice
# ============================================================================

"""
1. HIGHEST LEVEL: "Clear the dungeon"
   └── Clear current level
       ├── Explore all regions
       ├── Collect valuable items  
       ├── Find stairs
       └── Descend

2. HIGH LEVEL: "Explore all regions"
   └── For each unexplored region:
       ├── Plan path through regions (Room1 -> Corridor2 -> Room3)
       ├── Navigate between regions
       └── Explore within region

3. MID LEVEL: "Navigate between regions"
   └── Use region connectivity graph
       ├── Find doorways/connections
       ├── Handle obstacles (locked doors)
       └── Move to next region

4. LOW LEVEL: "Move within region"
   └── Local pathfinding (A*, Theta*, etc)
       ├── Avoid monsters if weak
       ├── Pick up items on the way
       └── Search for secrets

This mirrors human reasoning:
- "I need to explore this level" (high-level goal)
- "I'll check that room to the north" (region-based planning)  
- "I'll go through this corridor" (connectivity reasoning)
- "I'll walk around this pillar" (local pathfinding)
"""

print("Hierarchical NetHack reasoning system designed!")
print("\nKey concepts:")
print("1. Regions (rooms/corridors) as planning units")
print("2. Multi-level goal decomposition") 
print("3. Connectivity graph for navigation")
print("4. Context-appropriate pathfinding algorithms")
print("5. Human-like exploration strategies")