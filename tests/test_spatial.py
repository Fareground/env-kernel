"""Tests for kernel/spatial.py"""
from fg_env_kernel.spatial import NoSpace, GridSpace, GraphSpace, Continuous2DSpace


class TestNoSpace:
    def test_distance_always_zero(self):
        s = NoSpace()
        assert s.distance("a", "b") == 0.0

    def test_all_locations_valid(self):
        s = NoSpace()
        assert s.is_valid_location("anything") is True


class TestGridSpace:
    def test_distance(self):
        s = GridSpace(width=10, height=10)
        assert s.distance("0,0", "3,4") == 5.0  # 3-4-5 triangle
        assert s.distance("0,0", "0,0") == 0.0

    def test_adjacent(self):
        s = GridSpace(width=10, height=10)
        adj = s.adjacent("5,5")
        assert len(adj) == 8  # Center cell has 8 neighbors

        # Corner has 3
        adj_corner = s.adjacent("0,0")
        assert len(adj_corner) == 3

    def test_valid_location(self):
        s = GridSpace(width=5, height=5)
        assert s.is_valid_location("0,0") is True
        assert s.is_valid_location("4,4") is True
        assert s.is_valid_location("5,5") is False
        assert s.is_valid_location("invalid") is False

    def test_entities_within_range(self):
        s = GridSpace(width=10, height=10)
        locs = {"e1": "0,0", "e2": "1,1", "e3": "9,9"}
        nearby = s.entities_within_range("0,0", 2.0, locs)
        assert "e1" in nearby
        assert "e2" in nearby
        assert "e3" not in nearby


class TestGraphSpace:
    def test_basic(self):
        s = GraphSpace()
        s.add_node("tavern")
        s.add_node("market")
        s.add_node("castle")
        s.add_edge("tavern", "market", weight=1.0)
        s.add_edge("market", "castle", weight=3.0)

        assert s.distance("tavern", "market") == 1.0
        assert s.distance("tavern", "castle") == 4.0  # Shortest path: tavern→market→castle
        assert "market" in s.adjacent("tavern")
        assert "castle" not in s.adjacent("tavern")

    def test_disconnected_nodes(self):
        """Disconnected nodes should have infinite distance."""
        s = GraphSpace()
        s.add_node("island_a")
        s.add_node("island_b")
        # No edges between them
        assert s.distance("island_a", "island_b") == float('inf')

    def test_bidirectional(self):
        s = GraphSpace()
        s.add_node("a")
        s.add_node("b")
        s.add_edge("a", "b", weight=2.0, bidirectional=True)
        assert s.distance("a", "b") == 2.0
        assert s.distance("b", "a") == 2.0

    def test_is_valid(self):
        s = GraphSpace()
        s.add_node("tavern")
        assert s.is_valid_location("tavern") is True
        assert s.is_valid_location("dungeon") is False


class TestContinuous2DSpace:
    def test_distance(self):
        s = Continuous2DSpace(width=100, height=100)
        s.register_position("e1", 0.0, 0.0)
        s.register_position("e2", 3.0, 4.0)
        assert abs(s.distance("e1", "e2") - 5.0) < 0.001

    def test_bounds_clamping(self):
        s = Continuous2DSpace(width=100, height=100)
        s.register_position("e1", 150.0, -10.0)
        pos = s.get_position("e1")
        assert pos == (100.0, 0.0)

    def test_unknown_entity_infinite_distance(self):
        s = Continuous2DSpace()
        assert s.distance("unknown1", "unknown2") == float('inf')
