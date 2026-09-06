import pytest
from fg_env_kernel.pipeline.compile import compile_template
from fg_env_kernel.pipeline.loader import load_world
from fg_env_kernel.pipeline.smoke import smoke_test
from fg_env_kernel.phase_handlers import PHASE_HANDLER_REGISTRY, PhaseHandlerResult


def schema(handler):
    return {'name': 'Daily inventory', 'entity_types': [{'name': 'Stock', 'role': 'agent', 'properties': [{'name':'quantity','type':'int','default':10}]}],
            'entities': [{'id':'stock','name':'Stock','entity_type':'Stock'}],
            'temporal': {'phases':[{'name':'daily_update','handler':handler}]}}


def test_unknown_handler_fails_compile_at_the_authored_path():
    result = compile_template(schema('simulate_day'))
    assert not result.ok
    assert any(error.path == 'temporal.phases[0].handler' and 'simulate_day' in error.message for error in result.errors)


def test_legacy_unknown_handler_aborts_execution_and_smoke():
    state, engine = load_world(schema('simulate_day'))
    report = smoke_test(engine, rounds=3)
    assert not report.healthy
    assert not report.completed
    assert 'simulate_day' in report.crash
    assert state.entities['stock'].properties['quantity'] == 10
    assert any(event.event_type == 'phase_handler_error' for event in state.event_log.get_all())
    assert not any(event.event_type == 'round_end' for event in state.event_log.get_all())


def test_registered_handler_failure_never_continues_to_agent_actions(monkeypatch):
    class BrokenHandler:
        def execute(self, **kwargs): raise ValueError('Daily stock update failed')
    monkeypatch.setitem(PHASE_HANDLER_REGISTRY, 'test_daily_update', BrokenHandler())
    result = compile_template(schema('test_daily_update'))
    assert result.ok
    report = smoke_test(result.engine, rounds=3)
    assert not report.healthy
    assert 'Daily stock update failed' in report.crash


def test_registered_handler_runs_successfully(monkeypatch):
    class DailyHandler:
        def execute(self, *, state, **kwargs):
            state.entities['stock'].properties['quantity'] -= 1
            return PhaseHandlerResult(events=[{'type':'stock_used'}])
    monkeypatch.setitem(PHASE_HANDLER_REGISTRY, 'test_daily_update', DailyHandler())
    result = compile_template(schema('test_daily_update'))
    assert result.ok
    result.engine.max_rounds = 3
    result.engine.run()
    assert result.state.entities['stock'].properties['quantity'] == 7
