"""EventLog indexing and event-triggered termination (T-454).

``event_triggered`` used to rescan the whole event list once per elapsed round
on every check, so long games slowed super-linearly. These tests pin the
indexed replacement to the exact old semantics and prove the per-check cost no
longer grows with history.
"""

import copy

import pytest

from fg_env_kernel import termination as term
from fg_env_kernel.engine import TerminationCondition
from fg_env_kernel.event import EventLog, SimEvent
from fg_env_kernel.state import WorldState


def _event(event_type, round_number):
    return SimEvent(event_type=event_type, round_number=round_number, phase="p")


def _scan_count(log, event_type, current_round):
    """The pre-index definition: scan every round 1..current_round."""
    return sum(
        1
        for r in range(1, current_round + 1)
        for e in log.get_all()
        if e.round_number == r and e.event_type == event_type
    )


def _trigger(count, event_type="goal"):
    return TerminationCondition(
        name="goal_reached", check_type="event_triggered",
        params={"event_type": event_type, "count": count},
    )


def _first_firing_round(state, condition, script):
    """Play ``script`` (a list of per-round event types) and return the
    round on which ``condition`` first fires, or None."""
    for rnd, types in enumerate(script, start=1):
        state.temporal.current_round = rnd
        for etype in types:
            state.event_log.emit(_event(etype, rnd))
        if term.evaluate(state, condition):
            return rnd
    return None


class TestCountType:
    def test_matches_full_scan_across_rounds(self):
        log = EventLog()
        for rnd in range(0, 12):
            for etype in ("goal", "move", "goal") if rnd % 3 else ("move",):
                log.emit(_event(etype, rnd))
        for current in range(0, 14):
            for etype in ("goal", "move", "absent"):
                assert log.count_type(etype, current) == _scan_count(log, etype, current)

    def test_excludes_setup_future_and_fractional_rounds(self):
        log = EventLog()
        for rnd in (0, -1, None, 2.5, 3.0, 3, 7):
            log.emit(_event("goal", rnd))
        # Only 3.0 and 3 fall in 1..5; 7 is past the window.
        assert log.count_type("goal", 5) == 2
        assert log.count_type("goal", 7) == 3
        assert log.count_type("goal", 2) == 0

    def test_window_below_latest_round_after_rewind(self):
        log = EventLog()
        for rnd in range(1, 6):
            log.emit(_event("goal", rnd))
        # A check at an earlier round (e.g. after rewinding current_round)
        # must not see later rounds.
        assert log.count_type("goal", 2) == 2

    def test_get_round_order_and_isolation(self):
        log = EventLog()
        events = [_event(t, r) for r, t in [(1, "a"), (2, "b"), (1, "c"), (2, "d")]]
        for e in events:
            log.emit(e)
        assert log.get_round(1) == [events[0], events[2]]
        assert log.get_round(2) == [events[1], events[3]]
        assert log.get_round(9) == []
        log.get_round(1).clear()  # returned list is a copy
        assert len(log.get_round(1)) == 2


class TestTrimming:
    def test_counts_and_rounds_track_retained_events_only(self):
        log = EventLog(max_events=5)
        for rnd in range(1, 5):
            log.emit(_event("goal", rnd))
            log.emit(_event("move", rnd))
        # 8 emitted, 5 retained: rounds 2(move),3,3,4,4
        retained = log.get_all()
        assert [(e.round_number, e.event_type) for e in retained] == [
            (2, "move"), (3, "goal"), (3, "move"), (4, "goal"), (4, "move"),
        ]
        for etype in ("goal", "move"):
            for current in range(0, 6):
                assert log.count_type(etype, current) == _scan_count(log, etype, current)
        assert log.get_round(1) == []
        assert [e.event_type for e in log.get_round(2)] == ["move"]

    def test_type_fully_trimmed_counts_zero(self):
        log = EventLog(max_events=2)
        log.emit(_event("goal", 1))
        log.emit(_event("move", 2))
        log.emit(_event("move", 3))
        assert log.count_type("goal", 3) == 0
        assert log.get_round(1) == []

    def test_termination_fires_same_round_as_scan_under_trimming(self):
        script = [["goal"] + ["noise"] * 3 for _ in range(10)]
        state = WorldState()
        state.event_log = EventLog(max_events=6)
        condition = _trigger(count=3)
        fired = _first_firing_round(state, condition, script)
        # With 6 retained events (≤ 2 goals visible at once) the old scan
        # never reached 3 either — trimming semantics are unchanged.
        assert fired is None
        assert state.event_log.count_type("goal", 10) == _scan_count(state.event_log, "goal", 10)


class TestTerminationSemantics:
    @pytest.mark.parametrize("count,expected_round", [(1, 2), (2, 4), (3, 4), (4, 7), (5, None)])
    def test_fires_on_same_round_as_full_scan(self, count, expected_round):
        script = [["move"], ["goal"], ["move"], ["goal", "goal"], [], ["move"], ["goal"]]
        state = WorldState()
        # Pre-game setup event must not count, same as before.
        state.event_log.emit(_event("goal", 0))
        assert _first_firing_round(state, _trigger(count), script) == expected_round

    def test_missing_event_type_never_fires(self):
        state = WorldState()
        state.event_log.emit(_event("goal", 1))
        state.temporal.current_round = 1
        condition = TerminationCondition(
            name="bad", check_type="event_triggered", params={"count": 1})
        assert term.evaluate(state, condition) is False

    def test_restored_state_fires_on_same_round(self):
        """A state copied mid-game (as a restore/fork does) carries a
        consistent index and ends on the same round as the original."""
        script = [["move"], ["goal"], ["move"], ["goal"], ["goal"]]
        condition = _trigger(count=3)
        original = WorldState()
        assert _first_firing_round(original, condition, script[:2]) is None

        restored = copy.deepcopy(original)
        for state in (original, restored):
            fired = None
            for rnd, types in enumerate(script[2:], start=3):
                state.temporal.current_round = rnd
                for etype in types:
                    state.event_log.emit(_event(etype, rnd))
                if term.evaluate(state, condition):
                    fired = rnd
                    break
            assert fired == 5

    def test_rebuild_index_after_wholesale_replacement(self):
        log = EventLog()
        log.emit(_event("goal", 1))
        log._events = [_event("goal", 1), _event("goal", 2), _event("move", 2)]
        log._rebuild_index()
        assert log.count_type("goal", 2) == 2
        assert [e.event_type for e in log.get_round(2)] == ["goal", "move"]

    def test_runtime_legacy_path_uses_same_count(self):
        from fg_env_kernel.runtime.termination import _evaluate_condition

        class _Registry:
            class terminations:  # noqa: N801 — mimic registry attribute
                @staticmethod
                def has(_name):
                    return False

        class _Engine:
            registry = _Registry()
            _rng = None

        engine = _Engine()
        engine.state = WorldState()
        engine._evaluate_condition = lambda tc: _evaluate_condition(engine, tc)
        engine.state.event_log.emit(_event("goal", 0))
        engine.state.event_log.emit(_event("goal", 1))
        engine.state.temporal.current_round = 1
        assert _evaluate_condition(engine, _trigger(count=1)) is True
        assert _evaluate_condition(engine, _trigger(count=2)) is False


class TestCost:
    def test_check_does_not_read_whole_history(self, monkeypatch):
        """A check must not touch the event list, however long the game."""
        state = WorldState()
        condition = _trigger(count=10**9)
        reads = {"n": 0}
        real_get_round = EventLog.get_round
        real_get_all = EventLog.get_all

        def counting_get_round(self, r):
            reads["n"] += 1
            return real_get_round(self, r)

        def counting_get_all(self):
            reads["n"] += 1
            return real_get_all(self)

        monkeypatch.setattr(EventLog, "get_round", counting_get_round)
        monkeypatch.setattr(EventLog, "get_all", counting_get_all)
        rounds = 500
        for rnd in range(1, rounds + 1):
            state.temporal.current_round = rnd
            for etype in ("goal", "move", "move"):
                state.event_log.emit(_event(etype, rnd))
            assert term.evaluate(state, condition) is False
        # The old implementation made rounds*(rounds+1)/2 get_round scans.
        assert reads["n"] == 0
        assert state.event_log.count_type("goal", rounds) == rounds

    def test_per_check_work_is_flat_as_history_grows(self):
        """Time checks late in a long game against early ones — the old
        scan grew with (rounds x events); the index keeps it flat."""
        import time

        state = WorldState()
        condition = _trigger(count=10**9)
        samples = {}
        for rnd in range(1, 2001):
            state.temporal.current_round = rnd
            for etype in ("goal",) + ("move",) * 9:
                state.event_log.emit(_event(etype, rnd))
            if rnd in (100, 2000):
                start = time.perf_counter()
                for _ in range(200):
                    term.evaluate(state, condition)
                samples[rnd] = time.perf_counter() - start
        # 20x the history; a history scan would be ~400x slower. Allow a
        # generous margin for timer noise.
        assert samples[2000] < samples[100] * 5 + 0.01
