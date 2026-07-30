"""Spatial model implementations."""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class SpatialModel(ABC):
    """Abstract spatial interface. All spatial queries go through this."""

    @abstractmethod
    def distance(self, loc_a: str, loc_b: str) -> float:
        """Distance between two locations."""
        ...

    @abstractmethod
    def adjacent(self, location: str) -> List[str]:
        """Return locations adjacent/reachable from the given location."""
        ...

    @abstractmethod
    def all_locations(self) -> List[str]:
        """Return all valid location identifiers."""
        ...

    @abstractmethod
    def is_valid_location(self, location: str) -> bool:
        """Check if a location identifier is valid."""
        ...

    def entities_at(self, location: str, entity_locations: Dict[str, str]) -> List[str]:
        """Find all entity IDs at a given location."""
        return [eid for eid, loc in entity_locations.items() if loc == location]

    def entities_within_range(
        self, origin: str, max_distance: float, entity_locations: Dict[str, str]
    ) -> List[str]:
        """Find all entity IDs within range of a location."""
        result = []
        for eid, loc in entity_locations.items():
            if self.distance(origin, loc) <= max_distance:
                result.append(eid)
        return result


class NoSpace(SpatialModel):
    """No spatial model -- everything is co-located."""

    def distance(self, loc_a: str, loc_b: str) -> float:
        return 0.0

    def adjacent(self, location: str) -> List[str]:
        return []

    def all_locations(self) -> List[str]:
        return ["everywhere"]

    def is_valid_location(self, location: str) -> bool:
        return True


@dataclass
class GridSpace(SpatialModel):
    """2D grid spatial model. Locations are 'x,y' strings."""
    width: int = 10
    height: int = 10

    def _parse(self, loc: str) -> Tuple[int, int]:
        parts = loc.split(",")
        return int(parts[0]), int(parts[1])

    def distance(self, loc_a: str, loc_b: str) -> float:
        ax, ay = self._parse(loc_a)
        bx, by = self._parse(loc_b)
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def adjacent(self, location: str) -> List[str]:
        x, y = self._parse(location)
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append(f"{nx},{ny}")
        return neighbors

    def all_locations(self) -> List[str]:
        return [f"{x},{y}" for x in range(self.width) for y in range(self.height)]

    def is_valid_location(self, location: str) -> bool:
        try:
            x, y = self._parse(location)
            return 0 <= x < self.width and 0 <= y < self.height
        except (ValueError, IndexError):
            return False


@dataclass
class GraphSpace(SpatialModel):
    """
    Graph/node-based spatial model. Locations are named nodes.
    Edges define adjacency (optionally weighted for distance).
    """
    nodes: List[str] = field(default_factory=list)
    edges: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    def add_node(self, name: str):
        """Add a location node."""
        if name not in self.nodes:
            self.nodes.append(name)
            self.edges.setdefault(name, [])

    def add_edge(self, a: str, b: str, weight: float = 1.0, bidirectional: bool = True):
        """Add an edge between two nodes."""
        self.edges.setdefault(a, []).append((b, weight))
        if bidirectional:
            self.edges.setdefault(b, []).append((a, weight))

    def distance(self, loc_a: str, loc_b: str) -> float:
        """Shortest-path distance using Dijkstra's algorithm."""
        if loc_a == loc_b:
            return 0.0
        import heapq
        # Dijkstra's from loc_a
        dist: Dict[str, float] = {loc_a: 0.0}
        heap = [(0.0, loc_a)]
        visited: set = set()
        while heap:
            d, node = heapq.heappop(heap)
            if node == loc_b:
                return d
            if node in visited:
                continue
            visited.add(node)
            for neighbor, weight in self.edges.get(node, []):
                nd = d + weight
                if neighbor not in dist or nd < dist[neighbor]:
                    dist[neighbor] = nd
                    heapq.heappush(heap, (nd, neighbor))
        return float('inf')

    def adjacent(self, location: str) -> List[str]:
        return [node for node, _ in self.edges.get(location, [])]

    def all_locations(self) -> List[str]:
        return list(self.nodes)

    def is_valid_location(self, location: str) -> bool:
        return location in self.nodes


@dataclass
class Continuous2DSpace(SpatialModel):
    """Continuous 2D space. Entity IDs map to (x, y) positions."""
    width: float = 100.0
    height: float = 100.0
    _positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def register_position(self, entity_id: str, x: float, y: float):
        """Register or update an entity's position."""
        self._positions[entity_id] = (
            max(0.0, min(self.width, x)),
            max(0.0, min(self.height, y)),
        )

    def get_position(self, entity_id: str) -> Optional[Tuple[float, float]]:
        """Get an entity's position."""
        return self._positions.get(entity_id)

    def distance(self, loc_a: str, loc_b: str) -> float:
        if loc_a in self._positions and loc_b in self._positions:
            ax, ay = self._positions[loc_a]
            bx, by = self._positions[loc_b]
            return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        return float('inf')

    def adjacent(self, location: str) -> List[str]:
        return []  # Continuous space has no discrete adjacency

    def all_locations(self) -> List[str]:
        return list(self._positions.keys())

    def is_valid_location(self, location: str) -> bool:
        return location in self._positions
