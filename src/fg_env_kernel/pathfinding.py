"""Pathfinding on the location adjacency graph.

Provides A* shortest path, BFS reachability, and distance queries.
Used by the engine for movement validation and spatial reasoning.
"""
import heapq
from collections import deque
from typing import Dict, List, Optional


class Pathfinder:
    """Pathfinding utilities on a location adjacency graph."""

    @staticmethod
    def find_path(
        adjacency: Dict[str, List[str]],
        start: str,
        goal: str,
        costs: Optional[Dict[str, float]] = None,
    ) -> Optional[List[str]]:
        """A* shortest path. Returns [start, ..., goal] or None if unreachable.

        Args:
            adjacency: location_id -> list of adjacent location_ids
            start: starting location
            goal: destination location
            costs: optional {location_id: cost} for weighted pathfinding.
                   Default cost per step is 1.0.
        """
        if start == goal:
            return [start]

        if start not in adjacency and goal not in adjacency:
            return None

        # A* with uniform heuristic (degenerates to Dijkstra)
        # heap entries: (cost, location, path)
        open_set: list = [(0.0, start, [start])]
        visited = set()

        while open_set:
            cost, current, path = heapq.heappop(open_set)
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)

            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    step_cost = costs.get(neighbor, 1.0) if costs else 1.0
                    new_cost = cost + step_cost
                    heapq.heappush(open_set, (new_cost, neighbor, path + [neighbor]))

        return None

    @staticmethod
    def find_all_reachable(
        adjacency: Dict[str, List[str]],
        start: str,
        max_distance: int = -1,
    ) -> Dict[str, int]:
        """BFS: returns {location: distance} for all reachable locations from start.

        Args:
            adjacency: location_id -> list of adjacent location_ids
            start: starting location
            max_distance: maximum distance to explore. -1 = unlimited.

        Returns:
            Dict mapping location_id to distance from start.
            Includes start itself at distance 0.
        """
        distances: Dict[str, int] = {start: 0}
        queue: deque = deque([start])

        while queue:
            current = queue.popleft()
            current_dist = distances[current]

            if 0 <= max_distance <= current_dist:
                continue

            for neighbor in adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    @staticmethod
    def distance(
        adjacency: Dict[str, List[str]],
        start: str,
        goal: str,
    ) -> int:
        """Shortest distance (hop count) between two locations.

        Returns -1 if unreachable.
        """
        if start == goal:
            return 0

        visited = {start}
        queue: deque = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return -1

    @staticmethod
    def are_adjacent(
        adjacency: Dict[str, List[str]],
        loc_a: str,
        loc_b: str,
    ) -> bool:
        """Check if two locations are directly adjacent."""
        return loc_b in adjacency.get(loc_a, [])
