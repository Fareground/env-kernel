"""Unit tests for the continuous-time model's scheduling + serialization,
focused on the recurring environment-tick clock that drives physics."""
from fg_env_kernel import ContinuousTemporalModel


def test_is_empty_then_not():
    ct = ContinuousTemporalModel()
    assert ct.is_empty()
    ct.schedule_agent_turn("a")
    assert not ct.is_empty()


def test_environment_tick_schedules_at_interval():
    ct = ContinuousTemporalModel(environment_interval=2.5, max_time=100)
    ev = ct.schedule_environment_tick()
    assert ev is not None
    assert ev.fire_time == 2.5
    assert ev.event_type == "environment"
    assert ev.data.get("recurring") is True


def test_environment_tick_disabled_when_interval_zero():
    ct = ContinuousTemporalModel(environment_interval=0.0)
    assert ct.schedule_environment_tick() is None


def test_environment_tick_not_past_max_time():
    ct = ContinuousTemporalModel(environment_interval=5.0, max_time=3.0)
    assert ct.schedule_environment_tick() is None


def test_recurring_ticks_advance_in_order():
    ct = ContinuousTemporalModel(environment_interval=1.0, max_time=10)
    ct.schedule_environment_tick()
    times = []
    while True:
        ev = ct.pop_next_event()
        if ev is None:
            break
        times.append(ev.fire_time)
        if ev.data.get("recurring"):
            ct.schedule_environment_tick()  # reschedule like the engine does
    assert times == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def test_serialization_roundtrip_preserves_environment_interval():
    ct = ContinuousTemporalModel(
        action_durations={"trade": 0.5}, default_turn_interval=2.0,
        max_time=50.0, environment_interval=0.25,
    )
    ct.schedule_agent_turn("a", at_time=1.0)
    restored = ContinuousTemporalModel.from_dict(ct.to_dict())
    assert restored.environment_interval == 0.25
    assert restored.default_turn_interval == 2.0
    assert restored.action_durations == {"trade": 0.5}
    assert restored.max_time == 50.0
    assert not restored.is_empty()


def test_zero_action_duration_still_advances_the_clock():
    """A configured zero/negative action duration must not wedge the loop:
    every scheduled next-turn advances time by at least the minimum epsilon,
    so pop_next_event always eventually passes max_time."""
    ct = ContinuousTemporalModel(
        action_durations={"move": 0.0}, max_time=1.0, max_events=5000
    )
    assert ct.get_action_duration("move") >= ct.MIN_ACTION_DURATION
    ct.schedule_agent_turn("a", at_time=0.0)
    processed = 0
    last_time = -1.0
    while True:
        event = ct.pop_next_event()
        if event is None:
            break
        assert ct.current_time >= last_time  # time never goes backwards
        last_time = ct.current_time
        ct.schedule_next_turn_after_action("a", "move")
        processed += 1
        assert processed <= ct.max_events, "loop ran past its event backstop"
    # The backstop guarantees termination even though a zero duration would
    # otherwise need ~1e9 tiny steps to reach max_time.
    assert processed == ct.max_events


def test_negative_explicit_duration_is_floored():
    ct = ContinuousTemporalModel(max_time=5.0)
    ev = ct.schedule_next_turn_after_action("a", "act", action_duration=-3.0)
    assert ev.fire_time > ct.current_time
