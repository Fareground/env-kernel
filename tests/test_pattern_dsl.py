"""Pattern DSL tests — board pattern queries as expression functions.

This is the final piece of the kernel's declarative coverage. With
$legal_moves / $is_threatened / $is_in_check available as expression
functions, JSON can express:

  - "Can this piece move to that square?"  → $in($to, $legal_moves(...))
  - "Is my king in check?"                  → $is_in_check($actor.color)
  - "Is this square attacked by enemy?"     → $is_threatened($sq, "black")
  - "Is the path clear for the rook?"       → $line_clear($from, $to)

Together with $shortest_path / $connected_component on the general
adjacency graph, these close the gap for spatial/network games.
"""
import pytest

from fg_env_kernel.effects import resolve_expression
from fg_env_kernel.entity import Entity, EntityType
from fg_env_kernel.state import WorldState


# ---------------------------------------------------------------------------
# Build a board-backed world: 3x3 grid with marks at some cells
# (chess-class games would put pieces instead — same shape)
# ---------------------------------------------------------------------------


@pytest.fixture
def board_world():
    """A 3x3 board with X at (0,0) and O at (1,1)."""
    from fg_env_kernel.board_module import BoardModule
    from fg_env_kernel.domain_module import DomainModuleManager

    state = WorldState()
    state.entity_types["Player"] = EntityType(name="Player", role="agent")
    state.entities["x"] = Entity(id="x", name="X", entity_type="Player",
                                  properties={"mark": "X"})
    state.entities["o"] = Entity(id="o", name="O", entity_type="Player",
                                  properties={"mark": "O"})

    # Wire a BoardModule
    state.domain_modules = DomainModuleManager()
    board = BoardModule(
        name="board",
        params={"id": "main", "kind": "grid", "rows": 3, "cols": 3},
    )
    state.domain_modules.add_module(board)
    state.modules["domain_board"] = board

    # Place marks
    board.place_mark((0, 0), "X")
    board.place_mark((1, 1), "O")

    return state


# ---------------------------------------------------------------------------
# Basic board queries
# ---------------------------------------------------------------------------


class TestBoardQueries:
    def test_piece_at_returns_mark(self, board_world):
        # cell at (0,0) holds "X"
        val = resolve_expression("$piece_at([0, 0])", state=board_world)
        assert val == "X"

    def test_piece_at_empty(self, board_world):
        # (0, 1) is empty
        val = resolve_expression("$piece_at([0, 1])", state=board_world)
        assert val is None or val == ""

    def test_square_empty(self, board_world):
        assert resolve_expression("$square_empty([0, 1])", state=board_world) is True
        assert resolve_expression("$square_empty([0, 0])", state=board_world) is False

    def test_returns_safe_defaults_without_board(self):
        state = WorldState()
        # No board — graceful degradation
        assert resolve_expression("$is_threatened([0,0], white)", state=state) is False
        assert resolve_expression("$legal_moves(pawn, [0,0])", state=state) == []


# ---------------------------------------------------------------------------
# Graph algorithms — work without a board
# ---------------------------------------------------------------------------


class TestGraphAlgorithms:
    def test_shortest_path_linear(self):
        """A -- B -- C -- D"""
        state = WorldState()
        state.adjacency = {
            "A": ["B"], "B": ["A", "C"], "C": ["B", "D"], "D": ["C"],
        }
        path = resolve_expression("$shortest_path(A, D)", state=state)
        assert path == ["A", "B", "C", "D"]

    def test_shortest_path_disconnected(self):
        state = WorldState()
        state.adjacency = {"A": ["B"], "B": ["A"], "X": ["Y"], "Y": ["X"]}
        path = resolve_expression("$shortest_path(A, X)", state=state)
        assert path == []

    def test_shortest_path_same_node(self):
        state = WorldState()
        state.adjacency = {"A": []}
        path = resolve_expression("$shortest_path(A, A)", state=state)
        assert path == ["A"]

    def test_connected_component(self):
        """Two clusters: {A,B,C} and {X,Y}"""
        state = WorldState()
        state.adjacency = {
            "A": ["B"], "B": ["A", "C"], "C": ["B"],
            "X": ["Y"], "Y": ["X"],
        }
        comp_a = resolve_expression("$connected_component(A)", state=state)
        assert set(comp_a) == {"A", "B", "C"}
        comp_x = resolve_expression("$connected_component(X)", state=state)
        assert set(comp_x) == {"X", "Y"}

    def test_connected_component_size_is_longest_road_proxy(self):
        """Catan 'longest road' = largest connected road network.
        Compose with $len for a one-line metric."""
        state = WorldState()
        # Player_A owns roads forming a connected network of 4
        state.adjacency = {
            "road_1": ["road_2"], "road_2": ["road_1", "road_3"],
            "road_3": ["road_2", "road_4"], "road_4": ["road_3"],
        }
        # Predicate: agent owns a network of at least 3 roads
        size = resolve_expression(
            "$len($connected_component(road_1))", state=state
        )
        assert size == 4


# ---------------------------------------------------------------------------
# Composing pattern queries with predicates — the real win
# ---------------------------------------------------------------------------


class TestPatternsInsidePredicates:
    """The combinator that makes spatial games declarative: query
    functions composed inside expr strings."""

    def test_precondition_can_check_square_empty(self, board_world):
        """A move action's precondition: 'target square must be empty'."""
        from fg_env_kernel.predicates import evaluate
        # $params.target is [0, 1] (empty)
        expr = "$square_empty($params.target)"
        ok = evaluate(expr, params={"target": [0, 1]}, state=board_world)
        assert ok is True
        # Now target the occupied (0, 0)
        ok = evaluate(expr, params={"target": [0, 0]}, state=board_world)
        assert ok is False

    def test_termination_can_use_pattern_query(self, board_world):
        """Termination predicate: 'X has 3 in a row' — uses board_pattern,
        but we can also express it via custom expressions. Here: 'the
        board has more X than O' as an expr."""
        from fg_env_kernel.predicates import evaluate
        # Quick test that we can chain queries in predicates
        # — does (0,0) have an X mark? AND is (1,1) NOT empty?
        expr = "$piece_at([0,0]) == 'X' && !$square_empty([1,1])"
        assert evaluate(expr, state=board_world) is True
