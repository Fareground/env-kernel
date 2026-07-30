"""BoardModule — declarative grid / linear-ring boards (Phase 2).

Schema-only board mechanics: configure a grid or linear ring + the
move primitives each piece can use, and the kernel handles position
tracking, adjacency queries, line-clear checks, legal-move generation,
and pattern matching for win detection.

Replaces the ~3000 lines of bespoke board logic across battleship,
chess, monopoly, tic-tac-toe, backgammon, and dominoes.

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

Grid board (chess, tic-tac-toe, battleship, checkers):

    domain_modules:
      - name: board
        params:
          id: "main"
          kind: "grid"
          rows: 8
          cols: 8
          position_property: "position"          # entity prop: "row,col"
          side_property: "side"                  # for direction-aware pieces
          piece_property: "piece_type"           # entity prop: "pawn"/"knight"/...
          pieces:
            pawn:   { moves: ["forward_1", "forward_2_first", "diag_capture"], promote_at: "back_rank" }
            knight: { moves: ["l_shape"] }
            rook:   { moves: ["line_ortho"] }
            bishop: { moves: ["line_diag"] }
            queen:  { moves: ["line_ortho", "line_diag"] }
            king:   { moves: ["king_step"] }

Linear-ring board (monopoly, backgammon, race-tracks):

    domain_modules:
      - name: board
        params:
          id: "monopoly"
          kind: "linear_ring"
          spaces: 40
          position_property: "position"
          cells: [ {type: "go"}, {type: "property", group: "brown"}, ... ]
          # Movement on a ring is always dice_advance / advance_n.

═══════════════════════════════════════════════════════════════════════
WIN DETECTION
═══════════════════════════════════════════════════════════════════════

The `board_pattern` termination check_type already added in Phase 1
walks an entity's `board` property looking for row/col/diag runs.
For boards owned by BoardModule, we ALSO expose a snapshot grid the
WinPredicate can consult — see `BoardModule.snapshot_for_pattern()`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .domain_module import DomainModule

logger = logging.getLogger(__name__)

# Sentinel for distinguishing "cell never written" from "cell holds None".
_UNSET = object()


# A position is either (row, col) for a grid, or int for a linear ring.
Position = Union[Tuple[int, int], int]


# ───────────────────────────────────────────────────────────────────── #
# Moves DSL — declarative move primitives.
#
# Each primitive resolves a starting position + piece spec + board into
# a list of legal landing positions. Implementations live in the
# `_GridMoves` / `_RingMoves` namespaces below; the public list of
# names is what the agent / schema can use.
# ───────────────────────────────────────────────────────────────────── #

MOVE_PRIMITIVES: Tuple[str, ...] = (
    "forward_1",          # one step toward enemy's back rank
    "forward_2_first",    # 2 steps if still on starting row (clear path)
    "forward_n",          # sliding forward, blocked by any piece
    "back_1",             # one step toward own back rank
    "step_any_8",         # 1 step in any of 8 directions (king)
    "king_step",          # alias for step_any_8
    "l_shape",            # knight
    "line_ortho",         # sliding orthogonal (rook)
    "line_diag",          # sliding diagonal (bishop)
    "diag_capture",       # pawn diagonal, only captures
    "jump_over",          # checkers — diagonal jump over an opponent
    "dice_advance",       # linear board, advance by `dice_sum`
    # Tier 2 special moves — chess-style
    "castling",           # king + rook combined move (kingside / queenside)
    "en_passant",         # pawn diagonal capture of pawn that just doubled
    "bear_off",           # linear-ring: piece exits the board (backgammon)
)

# Recognised "side" values and their forward-direction vectors on a grid
# (row delta, col delta). Pieces without a side default to "white".
_SIDE_FORWARD: Dict[str, Tuple[int, int]] = {
    "white": (-1, 0),
    "black": (1, 0),
    "north": (-1, 0),
    "south": (1, 0),
    "east":  (0, 1),
    "west":  (0, -1),
}


# ───────────────────────────────────────────────────────────────────── #
# BoardModule
# ───────────────────────────────────────────────────────────────────── #

class BoardModule(DomainModule):
    """Declarative board with four topology kinds.

    Set ``kind`` in params to one of:

      "grid"        — 2D rows × cols (chess, checkers, tic-tac-toe, connect-4)
      "linear_ring" — circular track of N spaces (Monopoly, backgammon)
      "hex"         — axial-coordinate hex grid (Catan tiles, Settlers)
      "graph"       — arbitrary nodes + edges (Risk territories, network strategy,
                      Diplomacy provinces). Requires `nodes` (list of ids) +
                      `edges` (list of [from, to] pairs, undirected).

    All kinds support:
      - place_mark / mark_at / occupant_at
      - adjacency queries (via `adjacency(pos)` or expression
        `$adjacent_entities(pos)`)
      - movement primitives appropriate to the kind

    For graph kind specifically:
      - Adjacency comes from declared edges (not geometry)
      - Distance is BFS hops (`$distance` and `graph_distance` work)
      - Movement is "adjacent only" by default; use expression
        `$within_range(node, N)` for multi-hop reach
      - Most graph games (Risk, Catan) DON'T need a board at all —
        track territory ownership as entity properties and use
        `spatial: {"type": "graph", "nodes": [...], "edges": [...]}`
        at the world level for the same graph primitives globally.
    """

    def __init__(self, name: str = "board", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params or {})

        self._id: str = self._params.get("id", "main")
        self._kind: str = self._params.get("kind", "grid")
        self._rows: int = int(self._params.get("rows", 0))
        self._cols: int = int(self._params.get("cols", 0))
        self._spaces: int = int(self._params.get("spaces", 0))
        self._position_property: str = self._params.get("position_property", "position")
        self._side_property: str = self._params.get("side_property", "side")
        self._piece_property: str = self._params.get("piece_property", "piece_type")
        self._cells: List[Dict[str, Any]] = list(self._params.get("cells") or [])
        self._pieces: Dict[str, Dict[str, Any]] = dict(self._params.get("pieces") or {})
        # 6.P — Named zones for linear-ring boards: home, bar, bear-off.
        # Example for backgammon:
        #   "zones": {
        #     "home_white": { "range": [18, 23] },   # white's home board
        #     "home_black": { "range": [0, 5] },
        #     "bar":        { "sentinel": "bar" },   # off-grid holding zone
        #     "bear_off":   { "sentinel": "off" }    # off-grid completion
        #   }
        # Pieces with `position` ∈ a zone's range belong to that zone;
        # pieces with position == sentinel are in the off-grid zone.
        self._zones: Dict[str, Dict[str, Any]] = dict(self._params.get("zones") or {})

        if self._kind not in ("grid", "linear_ring", "hex", "graph"):
            raise ValueError(
                f"BoardModule.kind must be 'grid' | 'linear_ring' | 'hex' | 'graph', got {self._kind!r}"
            )
        if self._kind == "grid" and (self._rows <= 0 or self._cols <= 0):
            raise ValueError("Grid board requires rows>0 and cols>0")
        if self._kind == "linear_ring" and self._spaces <= 0:
            raise ValueError("linear_ring board requires spaces>0")
        # 5b — Hex board: axial coordinates (q, r). Cells form a rhombus
        # of `radius` from the origin OR a rectangle of rows×cols.
        if self._kind == "hex":
            if self._rows <= 0 or self._cols <= 0:
                raise ValueError("Hex board requires rows>0 and cols>0 (rectangular hex grid)")
        # 5b — Graph board: a `nodes` list of node ids + `edges` list of
        # [from, to] pairs (undirected by default).
        self._graph_nodes: List[str] = list(self._params.get("nodes") or [])
        self._graph_edges: List[List[str]] = list(self._params.get("edges") or [])
        if self._kind == "graph" and not self._graph_nodes:
            raise ValueError("Graph board requires non-empty 'nodes'")
        # Pre-build adjacency map for fast lookup
        self._graph_adj: Dict[str, List[str]] = {}
        for edge in self._graph_edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            a, b = str(edge[0]), str(edge[1])
            self._graph_adj.setdefault(a, []).append(b)
            self._graph_adj.setdefault(b, []).append(a)

        # Place-mark mode — cells hold mark VALUES directly (e.g., tic-
        # tac-toe 'X'/'O'), instead of (or in addition to) entity
        # positions. Engaged automatically when `PLACE_ON_BOARD` effects
        # are applied; persistent across the game; resets via `reset_cells`.
        self._cell_marks: Dict[Position, Any] = {}
        # Auto-mark assignment: when `auto_marks` is set on the board
        # (e.g. ["X", "O"] for tic-tac-toe, ["white", "black"] for chess),
        # the board deterministically assigns each unique actor the next
        # available mark in the list on their first place_mark call.
        # Eliminates the entire class of bugs where individual agents end
        # up without a `mark` property (late entity injection, missed
        # setup phase, etc.) — the board itself is the authority.
        self._auto_marks: List[Any] = list(self._params.get("auto_marks") or [])
        self._actor_marks: Dict[str, Any] = {}

    # ── Contract ──

    @property
    def description(self) -> str:
        if self._kind == "grid":
            return f"Grid board {self._rows}x{self._cols}"
        if self._kind == "hex":
            return f"Hex board {self._rows}x{self._cols} (axial coords)"
        if self._kind == "graph":
            return (f"Graph board with {len(self._graph_nodes)} nodes and "
                    f"{len(self._graph_edges)} edges (Risk/Catan-style topology)")
        return f"Linear ring board with {self._spaces} spaces"

    def tick(self, state: Any, round_number: int) -> List[Dict[str, Any]]:
        return []

    # ────────────────────────────────────────────────────────────── #
    # Position helpers
    # ────────────────────────────────────────────────────────────── #

    def position_of(self, entity: Any) -> Optional[Position]:
        """Read the entity's position property, normalised to (r,c) or int."""
        if entity is None:
            return None
        raw = entity.get(self._position_property) if hasattr(entity, "get") else None
        return self._parse_position(raw)

    def _parse_position(self, raw: Any) -> Optional[Position]:
        if raw is None:
            return None
        if self._kind == "linear_ring":
            try:
                return int(raw) % self._spaces
            except (TypeError, ValueError):
                return None
        # grid
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                return (int(raw[0]), int(raw[1]))
            except (TypeError, ValueError):
                return None
        if isinstance(raw, str) and "," in raw:
            try:
                r, c = raw.split(",", 1)
                return (int(r.strip()), int(c.strip()))
            except ValueError:
                return None
        return None

    def in_bounds(self, pos: Position) -> bool:
        if self._kind == "linear_ring":
            return isinstance(pos, int) and 0 <= pos < self._spaces
        if self._kind == "graph":
            return isinstance(pos, str) and pos in set(self._graph_nodes)
        if not (isinstance(pos, tuple) and len(pos) == 2):
            return False
        r, c = pos
        return 0 <= r < self._rows and 0 <= c < self._cols

    def cell_at(self, pos: Position) -> Optional[Dict[str, Any]]:
        """Linear-ring only: read the static cell config at this index."""
        if self._kind != "linear_ring":
            return None
        if not isinstance(pos, int) or not (0 <= pos < self._spaces):
            return None
        if pos < len(self._cells):
            return self._cells[pos]
        return None

    def entities_at(self, pos: Position, state: Any) -> List[Any]:
        """Walk every agent entity, return those whose position equals pos."""
        agents = list(state.get_agent_entities()) if hasattr(state, "get_agent_entities") else []
        out = []
        for ent in agents:
            if self.position_of(ent) == pos:
                out.append(ent)
        return out

    def occupant_at(self, pos: Position, state: Any) -> Optional[Any]:
        """First entity at pos (or None). Useful for grid games — most
        grids permit at most one piece per cell."""
        for ent in self.entities_at(pos, state):
            return ent
        return None

    # ────────────────────────────────────────────────────────────── #
    # Adjacency & line queries
    # ────────────────────────────────────────────────────────────── #

    def adjacency(self, pos: Position, diagonals: bool = True) -> List[Position]:
        """Neighbours. Grid: 4 or 8 directions. Ring: prev + next.
        Hex (axial q,r): 6 hex-neighbour offsets. Graph: declared edges."""
        if self._kind == "linear_ring":
            if not isinstance(pos, int):
                return []
            return [(pos - 1) % self._spaces, (pos + 1) % self._spaces]
        if self._kind == "graph":
            if not isinstance(pos, str):
                return []
            return list(self._graph_adj.get(pos, []))
        if self._kind == "hex":
            # Axial-coord hex neighbours: (+1,0), (-1,0), (0,+1), (0,-1), (+1,-1), (-1,+1)
            if not (isinstance(pos, tuple) and len(pos) == 2):
                return []
            q, r = pos
            offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
            return [(q + dq, r + dr) for dq, dr in offsets if self.in_bounds((q + dq, r + dr))]
        # grid
        if not (isinstance(pos, tuple) and len(pos) == 2):
            return []
        r, c = pos
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonals:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(r + dr, c + dc) for dr, dc in steps if self.in_bounds((r + dr, c + dc))]

    def graph_distance(self, a: Position, b: Position) -> Optional[int]:
        """BFS distance between two nodes on a graph board. Returns None
        if unreachable. Useful for Risk-style attack range checks."""
        if self._kind != "graph":
            return None
        if a == b:
            return 0
        from collections import deque
        seen = {a}
        q = deque([(a, 0)])
        while q:
            node, dist = q.popleft()
            for nbr in self._graph_adj.get(node, []):
                if nbr in seen:
                    continue
                if nbr == b:
                    return dist + 1
                seen.add(nbr)
                q.append((nbr, dist + 1))
        return None

    def hex_distance(self, a: Position, b: Position) -> Optional[int]:
        """Hex axial distance — `(|dq| + |dr| + |dq+dr|) / 2`."""
        if self._kind != "hex":
            return None
        if not (isinstance(a, tuple) and isinstance(b, tuple)):
            return None
        dq, dr = a[0] - b[0], a[1] - b[1]
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    def line_clear(self, a: Position, b: Position, state: Any) -> bool:
        """True if every cell strictly between `a` and `b` is empty AND
        they share a straight line (ortho or diag) on a grid. For
        linear_ring the "line" is just the arc; emptiness check applies."""
        if self._kind == "linear_ring":
            if not (isinstance(a, int) and isinstance(b, int)):
                return False
            # Walk the shorter arc forward.
            n = self._spaces
            cur = (a + 1) % n
            while cur != b:
                if self.occupant_at(cur, state) is not None:
                    return False
                cur = (cur + 1) % n
            return True
        if not (isinstance(a, tuple) and isinstance(b, tuple)):
            return False
        ar, ac = a; br, bc = b
        dr, dc = br - ar, bc - ac
        steps = max(abs(dr), abs(dc))
        if steps == 0:
            return True
        if abs(dr) not in (0, steps) or abs(dc) not in (0, steps):
            return False  # not a straight line
        sr = (dr // steps) if dr else 0
        sc = (dc // steps) if dc else 0
        for i in range(1, steps):
            if self.occupant_at((ar + sr * i, ac + sc * i), state) is not None:
                return False
        return True

    # ────────────────────────────────────────────────────────────── #
    # Legal moves — per-piece via the Moves DSL
    # ────────────────────────────────────────────────────────────── #

    def piece_spec(self, piece_type: str) -> Optional[Dict[str, Any]]:
        return self._pieces.get(piece_type)

    def valid_moves(
        self,
        entity: Any,
        state: Any,
        piece_type: Optional[str] = None,
    ) -> List[Position]:
        """Generate every legal landing position for this entity's piece
        from its current position, according to the piece's `moves` spec."""
        pos = self.position_of(entity)
        if pos is None:
            return []
        ptype = piece_type or (
            entity.get(self._piece_property) if hasattr(entity, "get") else None
        )
        spec = self.piece_spec(ptype) if ptype else None
        if not spec:
            # No piece spec → no constrained move set. Caller must
            # validate against engine-level effects (e.g. simple
            # set_position with their own bounds).
            return []
        side = entity.get(self._side_property) if hasattr(entity, "get") else None
        moves: List[Position] = []
        for prim in spec.get("moves", []):
            moves.extend(self._dispatch_move(prim, pos, side, entity, state, spec))
        # Dedupe while preserving order
        seen = set()
        out: List[Position] = []
        for m in moves:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _dispatch_move(
        self,
        prim: str,
        pos: Position,
        side: Optional[str],
        entity: Any,
        state: Any,
        spec: Dict[str, Any],
    ) -> Iterator[Position]:
        """Route a move primitive to its implementation."""
        if self._kind == "grid":
            return _GridMoves.generate(self, prim, pos, side, entity, state, spec)
        return _RingMoves.generate(self, prim, pos, side, entity, state, spec)

    # ────────────────────────────────────────────────────────────── #
    # Snapshot for `board_pattern` WinPredicate
    # ────────────────────────────────────────────────────────────── #

    # ────────────────────────────────────────────────────────────── #
    # Place-mark mode (tic-tac-toe, connect-four, gomoku)
    # ────────────────────────────────────────────────────────────── #

    def place_mark(self, pos: Position, mark: Any, overwrite: bool = False) -> bool:
        """Drop a mark value at `pos`. Returns True if placed, False if
        the cell was already taken (and overwrite=False).

        Treats `None`-mark cells as OCCUPIED (so a stray null placement
        can't be silently overwritten by the next player). The engine's
        PLACE_ON_BOARD handler uses `mark_for_actor()` to ensure marks
        are never null in the first place — this is defense in depth."""
        if not self.in_bounds(pos):
            return False
        existing = self._cell_marks.get(pos, _UNSET)
        if not overwrite and existing is not _UNSET:
            return False
        self._cell_marks[pos] = mark
        return True

    def mark_for_actor(self, actor_id: str) -> Optional[Any]:
        """Resolve a deterministic mark for `actor_id` using `auto_marks`.

        First unique actor gets `auto_marks[0]`, second gets `auto_marks[1]`,
        etc. The assignment is sticky — once an actor has a mark, they
        keep it for the rest of the game. Returns None when `auto_marks`
        isn't configured on this board (caller falls back to its own mark
        resolution).
        """
        if not self._auto_marks:
            return None
        if actor_id in self._actor_marks:
            return self._actor_marks[actor_id]
        slot = len(self._actor_marks) % len(self._auto_marks)
        mark = self._auto_marks[slot]
        self._actor_marks[actor_id] = mark
        return mark

    @property
    def actor_marks(self) -> Dict[str, Any]:
        """Read-only view of every actor's auto-assigned mark."""
        return dict(self._actor_marks)

    def check_for_win_pattern(
        self, patterns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return {mark, pattern, cells} if any placed mark forms a
        winning N-in-a-row run on the grid, else None.

        Used to emit `board_pattern_matched` events the moment a
        placement creates a winning line — without ending the sim.
        Lets best-of-N formats reset the board and continue."""
        if self._kind != "grid":
            return None
        if not self._cell_marks:
            return None
        patterns = patterns or ["row_3", "col_3", "diag_3"]
        # Build a 2-D snapshot of the place-marks only (skip entity
        # occupancy — pattern detection is about the marks).
        grid: List[List[Any]] = [[None] * self._cols for _ in range(self._rows)]
        for (r, c), v in self._cell_marks.items():
            if 0 <= r < self._rows and 0 <= c < self._cols:
                grid[r][c] = v
        rows, cols = self._rows, self._cols
        for pat in patterns:
            try:
                kind, n_s = pat.rsplit("_", 1)
                n = int(n_s)
            except ValueError:
                continue

            def _scan(seq: List[Any]) -> Optional[Dict[str, Any]]:
                for i in range(len(seq) - n + 1):
                    window = seq[i : i + n]
                    if any(v in (None, "", 0) for v in window):
                        continue
                    first = window[0]
                    if all(v == first for v in window):
                        return {"mark": first, "window": window}
                return None

            if kind == "row":
                for r in range(rows):
                    hit = _scan(list(grid[r]))
                    if hit:
                        cells = [[r, i] for i in range(cols) if grid[r][i] == hit["mark"]]
                        return {"mark": hit["mark"], "pattern": pat, "cells": cells}
            elif kind == "col":
                for c in range(cols):
                    seq = [grid[r][c] for r in range(rows)]
                    hit = _scan(seq)
                    if hit:
                        cells = [[i, c] for i in range(rows) if grid[i][c] == hit["mark"]]
                        return {"mark": hit["mark"], "pattern": pat, "cells": cells}
            elif kind == "diag":
                for r0 in range(rows - n + 1):
                    for c0 in range(cols - n + 1):
                        main_diag = [grid[r0 + i][c0 + i] for i in range(n)]
                        hit = _scan(main_diag)
                        if hit:
                            cells = [[r0 + i, c0 + i] for i in range(n)]
                            return {"mark": hit["mark"], "pattern": pat, "cells": cells}
                        anti_diag = [grid[r0 + i][c0 + n - 1 - i] for i in range(n)]
                        hit = _scan(anti_diag)
                        if hit:
                            cells = [[r0 + i, c0 + n - 1 - i] for i in range(n)]
                            return {"mark": hit["mark"], "pattern": pat, "cells": cells}
        return None

    def mark_at(self, pos: Position) -> Any:
        return self._cell_marks.get(pos)

    def empty_cells(self) -> List[Position]:
        """All in-bounds positions with no mark."""
        if self._kind == "grid":
            out: List[Position] = []
            for r in range(self._rows):
                for c in range(self._cols):
                    if (r, c) not in self._cell_marks:
                        out.append((r, c))
            return out
        return [i for i in range(self._spaces) if i not in self._cell_marks]

    def reset_cells(self) -> None:
        self._cell_marks.clear()

    def snapshot_grid(self, state: Any) -> List[List[Any]]:
        """For grid boards: render the current occupancy as a 2-D list.

        Two sources are merged, place-marks first so place-mark mode
        works standalone:
          (1) `self._cell_marks` — values dropped via `place_mark` /
              the PLACE_ON_BOARD effect (tic-tac-toe, connect-four).
          (2) Entity occupancy — pieces sitting on the board (chess).
        Each cell becomes the entity's `mark` / `side` / piece_type.
        """
        if self._kind != "grid":
            return []
        grid: List[List[Any]] = [[None] * self._cols for _ in range(self._rows)]

        # Place-marks first.
        for (r, c), mark in self._cell_marks.items():
            if 0 <= r < self._rows and 0 <= c < self._cols:
                grid[r][c] = mark

        # Then occupancy (overwrites empty cells; doesn't clobber marks).
        agents = list(state.get_agent_entities()) if hasattr(state, "get_agent_entities") else []
        for ent in agents:
            p = self.position_of(ent)
            if p is None or not self.in_bounds(p):
                continue
            r, c = p
            if grid[r][c] is not None:
                continue
            mark = None
            for prop in ("mark", "side", self._piece_property):
                try:
                    val = ent.get(prop)
                except Exception:
                    val = None
                if val:
                    mark = val
                    break
            grid[r][c] = mark or ent.id
        return grid

    # ────────────────────────────────────────────────────────────── #
    # Checkmate detection (6.M)
    # ────────────────────────────────────────────────────────────── #

    def is_square_attacked(self, square: Position, by_side: str, state: Any) -> bool:
        """Does any piece of `by_side` threaten `square`?

        Walks every enemy piece's `valid_moves`. Used for check/
        checkmate detection. Cheap enough for chess (≤16 pieces).
        """
        if self._kind != "grid":
            return False
        for ent in state.get_agent_entities():
            try:
                if ent.get(self._side_property) != by_side:
                    continue
                ptype = ent.get(self._piece_property)
            except Exception:
                continue
            if not ptype:
                continue
            for m in self.valid_moves(ent, state, piece_type=ptype):
                if m == square:
                    return True
        return False

    def is_in_check(self, king_side: str, state: Any) -> bool:
        """Is `king_side`'s king currently attacked?"""
        king_pos = None
        for ent in state.get_agent_entities():
            try:
                if (ent.get(self._side_property) == king_side
                        and ent.get(self._piece_property) in ("king", "King")):
                    king_pos = self.position_of(ent)
                    break
            except Exception:
                continue
        if king_pos is None:
            return False
        opp_side = self._opposite_side_of(king_side)
        return self.is_square_attacked(king_pos, opp_side, state)

    def is_in_checkmate(self, side: str, state: Any) -> bool:
        """`side` is in check AND has no legal move that escapes it.

        Tries every legal move of every `side` piece; for each, simulates
        the move + checks if `side` is still in check. If no move
        escapes, it's mate.
        """
        if not self.is_in_check(side, state):
            return False
        for ent in state.get_agent_entities():
            try:
                if ent.get(self._side_property) != side:
                    continue
            except Exception:
                continue
            from_pos = self.position_of(ent)
            if from_pos is None:
                continue
            for to_pos in self.valid_moves(ent, state):
                # Simulate by temporarily moving the piece + clearing any
                # piece on the target square, then restoring.
                captured_ent = self.occupant_at(to_pos, state) if isinstance(to_pos, tuple) else None
                captured_pos = None
                if captured_ent is not None and captured_ent is not ent:
                    captured_pos = self.position_of(captured_ent)
                    captured_ent.set(self._position_property, "_offboard")
                old_pos = from_pos
                ent.set(self._position_property, f"{to_pos[0]},{to_pos[1]}"
                        if isinstance(to_pos, tuple) else str(to_pos))
                still_in_check = self.is_in_check(side, state)
                # Restore
                ent.set(self._position_property,
                        f"{old_pos[0]},{old_pos[1]}"
                        if isinstance(old_pos, tuple) else str(old_pos))
                if captured_ent is not None and captured_pos is not None:
                    captured_ent.set(
                        self._position_property,
                        f"{captured_pos[0]},{captured_pos[1]}"
                        if isinstance(captured_pos, tuple) else str(captured_pos),
                    )
                if not still_in_check:
                    return False
        return True

    @staticmethod
    def _opposite_side_of(side: str) -> str:
        if side == "white": return "black"
        if side == "black": return "white"
        if side == "north": return "south"
        if side == "south": return "north"
        if side == "east":  return "west"
        if side == "west":  return "east"
        return side

    # ────────────────────────────────────────────────────────────── #
    # Perception — surface useful data for the LLM
    # ────────────────────────────────────────────────────────────── #

    def get_perception_data(self, entity_id: str, state: Any) -> Dict[str, Any]:
        try:
            ent = state.get_entity(entity_id) if hasattr(state, "get_entity") else None
        except Exception:
            ent = None
        data: Dict[str, Any] = {
            "board_id": self._id,
            "kind": self._kind,
        }
        if self._kind == "grid":
            data.update({
                "rows": self._rows,
                "cols": self._cols,
                "snapshot": self.snapshot_grid(state),
            })
        else:
            data.update({
                "spaces": self._spaces,
                "your_position": self.position_of(ent) if ent else None,
            })
        if ent is not None:
            moves = self.valid_moves(ent, state)
            if moves:
                data["your_legal_moves"] = [list(m) if isinstance(m, tuple) else m for m in moves]
        return data


# ───────────────────────────────────────────────────────────────────── #
# Grid moves — each method yields legal landing positions.
# ───────────────────────────────────────────────────────────────────── #

class _GridMoves:
    @staticmethod
    def generate(
        board: BoardModule, prim: str, pos: Position, side: Optional[str],
        entity: Any, state: Any, spec: Dict[str, Any],
    ) -> Iterator[Position]:
        method = getattr(_GridMoves, prim, None)
        if method is None:
            return iter(())
        return method(board, pos, side, entity, state, spec)

    @staticmethod
    def _forward(side: Optional[str]) -> Tuple[int, int]:
        if side and side in _SIDE_FORWARD:
            return _SIDE_FORWARD[side]
        return _SIDE_FORWARD["white"]

    @staticmethod
    def forward_1(board, pos, side, entity, state, spec) -> Iterator[Position]:
        dr, dc = _GridMoves._forward(side)
        r, c = pos
        dest = (r + dr, c + dc)
        if board.in_bounds(dest) and board.occupant_at(dest, state) is None:
            yield dest

    @staticmethod
    def forward_2_first(board, pos, side, entity, state, spec) -> Iterator[Position]:
        dr, dc = _GridMoves._forward(side)
        r, c = pos
        # Define "starting row" by side; configurable via spec.
        start_row = spec.get("starting_row")
        if start_row is None:
            start_row = board._rows - 2 if side in (None, "white", "north") else 1
        if r != start_row:
            return
        one = (r + dr, c + dc)
        two = (r + 2 * dr, c + 2 * dc)
        if (board.in_bounds(one) and board.occupant_at(one, state) is None
                and board.in_bounds(two) and board.occupant_at(two, state) is None):
            yield two

    @staticmethod
    def forward_n(board, pos, side, entity, state, spec) -> Iterator[Position]:
        dr, dc = _GridMoves._forward(side)
        r, c = pos
        n = int(spec.get("max_steps", max(board._rows, board._cols)))
        for i in range(1, n + 1):
            dest = (r + dr * i, c + dc * i)
            if not board.in_bounds(dest):
                return
            occ = board.occupant_at(dest, state)
            if occ is not None:
                # Allow capturing the blocker if its side differs.
                if _opposite_side(occ, entity, board):
                    yield dest
                return
            yield dest

    @staticmethod
    def back_1(board, pos, side, entity, state, spec) -> Iterator[Position]:
        dr, dc = _GridMoves._forward(side)
        r, c = pos
        dest = (r - dr, c - dc)
        if board.in_bounds(dest) and board.occupant_at(dest, state) is None:
            yield dest

    @staticmethod
    def step_any_8(board, pos, side, entity, state, spec) -> Iterator[Position]:
        r, c = pos
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            dest = (r + dr, c + dc)
            if not board.in_bounds(dest):
                continue
            occ = board.occupant_at(dest, state)
            if occ is None or _opposite_side(occ, entity, board):
                yield dest

    # Alias
    king_step = step_any_8

    @staticmethod
    def l_shape(board, pos, side, entity, state, spec) -> Iterator[Position]:
        r, c = pos
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]:
            dest = (r + dr, c + dc)
            if not board.in_bounds(dest):
                continue
            occ = board.occupant_at(dest, state)
            if occ is None or _opposite_side(occ, entity, board):
                yield dest

    @staticmethod
    def line_ortho(board, pos, side, entity, state, spec) -> Iterator[Position]:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            yield from _sliding(board, pos, dr, dc, entity, state)

    @staticmethod
    def line_diag(board, pos, side, entity, state, spec) -> Iterator[Position]:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            yield from _sliding(board, pos, dr, dc, entity, state)

    @staticmethod
    def diag_capture(board, pos, side, entity, state, spec) -> Iterator[Position]:
        dr, _ = _GridMoves._forward(side)
        r, c = pos
        for dc in (-1, 1):
            dest = (r + dr, c + dc)
            if not board.in_bounds(dest):
                continue
            occ = board.occupant_at(dest, state)
            if occ is not None and _opposite_side(occ, entity, board):
                yield dest

    @staticmethod
    def jump_over(board, pos, side, entity, state, spec) -> Iterator[Position]:
        # Checkers: 2 diagonal steps, must jump exactly one opponent.
        r, c = pos
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            mid = (r + dr, c + dc)
            land = (r + 2 * dr, c + 2 * dc)
            if not board.in_bounds(land):
                continue
            mid_occ = board.occupant_at(mid, state)
            land_occ = board.occupant_at(land, state)
            if mid_occ is not None and _opposite_side(mid_occ, entity, board) and land_occ is None:
                yield land

    # ── Tier-2 special moves ──

    @staticmethod
    def castling(board, pos, side, entity, state, spec) -> Iterator[Position]:
        """King-side / queen-side castling. Geometry only:
          - entity must NOT have moved (`has_moved` property false)
          - the rook on that side must NOT have moved
          - cells between king and rook must be empty
          - check / pass-through-check rules are out of scope here
            (combine with a Tier-3 checkmate predicate for full
             enforcement).
        King yields the LANDING square; the caller separately moves
        the rook via a partner effect (see kernel_recipes)."""
        r, c = pos
        if entity is None or hasattr(entity, "get") and bool(entity.get("has_moved")):
            return
        # Try both sides. The rook's column is taken from spec or
        # defaults to col 0 (queenside) / col 7 (kingside) for an 8×8 board.
        sides = spec.get("castling_sides") or [
            {"rook_col": 0, "land_col": c - 2},  # queenside
            {"rook_col": board._cols - 1, "land_col": c + 2},  # kingside
        ]
        for side_spec in sides:
            rook_col = int(side_spec.get("rook_col"))
            land_col = int(side_spec.get("land_col"))
            # Find rook
            rook_ent = board.occupant_at((r, rook_col), state)
            if rook_ent is None:
                continue
            try:
                if rook_ent.get(board._piece_property) not in ("rook", "Rook"):
                    continue
                if bool(rook_ent.get("has_moved")):
                    continue
            except Exception:
                continue
            # Cells strictly between king and rook must be empty
            c_lo, c_hi = min(c, rook_col), max(c, rook_col)
            clear = all(
                board.occupant_at((r, cc), state) is None
                for cc in range(c_lo + 1, c_hi)
            )
            if clear and 0 <= land_col < board._cols:
                yield (r, land_col)

    @staticmethod
    def en_passant(board, pos, side, entity, state, spec) -> Iterator[Position]:
        """Pawn en-passant capture. Reads the LAST emitted event of
        type `track_advanced` or `pawn_double_step` to find a pawn
        that just moved two squares; yields the diagonal landing
        square behind it.

        For the kernel-generic version we use a simpler rule: yield
        the diagonal-1 forward square ONLY when an opposite-side
        pawn sits next to us on the same rank. The engine sets up
        the "just-double-stepped" flag via the action's effect
        (e.g., `set has_double_stepped true` for one round)."""
        if entity is None:
            return
        dr, _ = _GridMoves._forward(side)
        r, c = pos
        for dc in (-1, 1):
            neighbour_pos = (r, c + dc)
            if not board.in_bounds(neighbour_pos):
                continue
            neighbour = board.occupant_at(neighbour_pos, state)
            if neighbour is None:
                continue
            if not _opposite_side(neighbour, entity, board):
                continue
            try:
                ptype = neighbour.get(board._piece_property)
                if ptype not in ("pawn", "Pawn"):
                    continue
                if not bool(neighbour.get("has_double_stepped")):
                    continue
            except Exception:
                continue
            dest = (r + dr, c + dc)
            if board.in_bounds(dest) and board.occupant_at(dest, state) is None:
                yield dest


def _sliding(
    board: BoardModule, pos: Position, dr: int, dc: int, entity: Any, state: Any,
) -> Iterator[Position]:
    """Slide in (dr, dc) until off-board or blocked."""
    r, c = pos
    step = 1
    while True:
        dest = (r + dr * step, c + dc * step)
        if not board.in_bounds(dest):
            return
        occ = board.occupant_at(dest, state)
        if occ is None:
            yield dest
            step += 1
            continue
        if _opposite_side(occ, entity, board):
            yield dest
        return


def _opposite_side(a: Any, b: Any, board: BoardModule) -> bool:
    if a is None or b is None:
        return False
    side_prop = board._side_property
    try:
        sa = a.get(side_prop)
        sb = b.get(side_prop)
    except Exception:
        return False
    return sa is not None and sb is not None and sa != sb


# ───────────────────────────────────────────────────────────────────── #
# Linear-ring moves
# ───────────────────────────────────────────────────────────────────── #

class _RingMoves:
    @staticmethod
    def generate(
        board: BoardModule, prim: str, pos: Position, side: Optional[str],
        entity: Any, state: Any, spec: Dict[str, Any],
    ) -> Iterator[Position]:
        method = getattr(_RingMoves, prim, None)
        if method is None:
            return iter(())
        return method(board, pos, side, entity, state, spec)

    @staticmethod
    def dice_advance(board, pos, side, entity, state, spec) -> Iterator[Position]:
        # Reads `dice_sum` from the entity property OR the most recent
        # event with a `dice_sum` field. Resolution happens at action
        # time; for `valid_moves` we offer the canonical 1..max range.
        max_steps = int(spec.get("max_steps", 12))
        for k in range(1, max_steps + 1):
            yield (pos + k) % board._spaces

    @staticmethod
    def forward_1(board, pos, side, entity, state, spec) -> Iterator[Position]:
        yield (pos + 1) % board._spaces

    @staticmethod
    def back_1(board, pos, side, entity, state, spec) -> Iterator[Position]:
        yield (pos - 1) % board._spaces

    @staticmethod
    def bear_off(board, pos, side, entity, state, spec) -> Iterator[Position]:
        """Linear-ring zone exit — piece leaves the board.

        Schema-side: a configured `bear_off_zone` on the board (or
        from spec.bear_off_from) names the contiguous range of cells
        from which a piece can bear off. Yields the sentinel position
        `-1` to indicate "off the board"; the effect handler
        consuming this should remove the piece from `position_property`
        or set a `borne_off` flag.
        """
        zone_range = spec.get("bear_off_from") or board._params.get("bear_off_zone")
        if not zone_range or len(zone_range) != 2:
            return
        lo, hi = int(zone_range[0]), int(zone_range[1])
        if lo <= pos <= hi:
            yield -1  # sentinel — caller marks piece as borne-off


__all__ = ["BoardModule", "MOVE_PRIMITIVES"]
