"""Settlement emitted by a domain tick ends execution before the next decision."""
from fg_env_kernel.domain.base import DomainModuleManager
from test_unbounded_execution import engine
from fg_env_kernel.engine import TerminationCondition


def test_domain_settlement_prevents_extra_decisions_and_marks_callback_terminal(monkeypatch):
    sim = engine()
    sim.termination_conditions = [TerminationCondition('settled', check_type='event_triggered',
        params={'event_type': 'match_over'})]
    sim.state.domain_modules = DomainModuleManager()
    monkeypatch.setattr(sim.state.domain_modules, 'tick_all', lambda state, round_num:
        [{'event_type': 'match_over', 'data': {'ticks_played': 4}}] if round_num == 5 else [])
    ended = []
    sim.on_round_end = lambda round_num, state: ended.append((round_num, sim.terminated_by))
    sim.run()
    assert sim.state.get_entity('a').properties['score'] == 4
    assert sim.terminated_by == 'settled'
    assert ended[-1] == (5, 'settled')
    events = sim.state.event_log.get_all()
    verdict = next(i for i, event in enumerate(events) if event.event_type == 'match_over')
    assert not any(event.event_type in {'action_attempted', 'action_resolved'} for event in events[verdict + 1:])
    assert sum(event.event_type == 'simulation_terminated' for event in events) == 1
