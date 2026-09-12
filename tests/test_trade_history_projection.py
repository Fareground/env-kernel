"""Perception keeps exact trade totals without rescanning old transcript rows."""
from types import SimpleNamespace

import pytest

from fg_env_kernel.event import EventLog, SimEvent
from fg_env_kernel.runtime.perception import _add_trade_history


def trade(actor='a', amount=3.1, action='buy', round_number=1):
    return SimEvent('action_resolved', round_number, 'trade', actor_id=actor,
                    action_name=action, data={'details': {'amount': amount, 'shares': 2, 'execution_price': 1.55}})


def view(state, actor='a'):
    result = {}
    _add_trade_history(state, actor, result)
    return result.get('trade_history')


def reference(log, actor):
    rows = []
    for event in log.get_all():
        if event.actor_id != actor or event.event_type != 'action_resolved' or event.action_name not in {'buy', 'sell', 'buy_yes', 'buy_no', 'sell_yes', 'sell_no'}:
            continue
        details = event.data.get('details', {})
        if not details.get('shares') and not details.get('amount'):
            continue
        rows.append({'round': event.round_number, 'action': event.action_name,
                     'amount': details.get('amount', 0), 'shares': details.get('shares', 0),
                     'price_at_trade': details.get('price_after') or details.get('execution_price') or details.get('new_price', 0)})
    if not rows:
        return None
    spent = sum(row['amount'] for row in rows if row['action'].startswith('buy'))
    received = sum(row['amount'] for row in rows if row['action'].startswith('sell'))
    return {'total_trades': len(rows), 'total_spent': round(spent, 2),
            'total_received': round(received, 2), 'realized_pnl': round(received-spent, 2),
            'recent_trades': rows[-10:]}


@pytest.mark.parametrize('cap', [None, 23])
def test_interleaved_history_matches_full_scan_with_trimming_and_out_of_order_rounds(cap):
    log = EventLog(max_events=cap)
    state = SimpleNamespace(event_log=log)
    for i in range(130):
        log.emit(trade('a' if i % 2 else 'b', i * .13, 'buy_yes' if i % 3 else 'sell_no', 130-i))
        log.emit(SimEvent('round_end', i, 'end'))
        for actor in ('a', 'b', 'unseen'):
            assert view(state, actor) == reference(log, actor)
    assert len(log.get_all()) == (cap or 260)


def test_append_reads_never_revisit_old_events_and_totals_do_not_truncate():
    log = EventLog()
    state = SimpleNamespace(event_log=log)
    for i in range(10000):
        log.emit(trade(amount=1, round_number=i))
    assert view(state)['total_trades'] == 10000
    class SuffixOnly(list):
        def __iter__(self):
            raise AssertionError('old transcript scan')
        def __getitem__(self, item):
            assert isinstance(item, slice) and item.start >= 10000
            return super().__getitem__(item)
    log._events = SuffixOnly(log._events)
    log.emit(trade(amount=7, action='sell', round_number=10001))
    summary = view(state)
    assert summary['total_trades'] == 10001
    assert summary['total_spent'] == 10000
    assert summary['total_received'] == 7
    assert len(summary['recent_trades']) == 10
    summary['recent_trades'][0]['amount'] = 99999
    assert view(state)['recent_trades'][0]['amount'] == 1


def test_rebuild_replacement_and_another_world_do_not_reuse_stale_totals():
    log = EventLog()
    state = SimpleNamespace(event_log=log)
    log.emit(trade(amount=1))
    assert view(state)['total_spent'] == 1
    log._events = [trade(amount=17)]
    log._rebuild_index()
    assert view(state)['total_spent'] == 17
    replacement = EventLog()
    replacement.emit(trade(amount=4))
    state.event_log = replacement
    assert view(state)['total_spent'] == 4
    assert view(SimpleNamespace(event_log=log))['total_spent'] == 17


def test_malformed_trade_does_not_break_other_actor_or_publish_partial_totals():
    log = EventLog(max_events=2)
    state = SimpleNamespace(event_log=log)
    log.emit(trade(amount='invalid'))
    log.emit(trade('b', amount=9))
    assert view(state) is None
    assert view(state, 'b')['total_spent'] == 9
    log.emit(trade(amount=5))
    assert view(state)['total_spent'] == 5


def test_real_checkpoint_restore_rebuilds_totals_not_later_cached_state():
    from test_execution_checkpoint import make_engine
    source = make_engine()
    source.step()
    source.state.event_log.emit(trade(amount=17))
    assert view(source.state)['total_spent'] == 17
    checkpoint = source.checkpoint()
    assert '_trade_history_projection' not in checkpoint['world']
    target = make_engine()
    target.state.event_log.emit(trade(amount=999))
    assert view(target.state)['total_spent'] == 999
    target.restore_checkpoint(checkpoint)
    assert view(target.state)['total_spent'] == 17
    target.state.event_log.emit(trade(amount=2))
    assert view(target.state)['total_spent'] == 19
    assert view(source.state)['total_spent'] == 17


def test_event_cursor_is_scoped_to_exact_log_and_reset_on_trim():
    first = EventLog(max_events=1)
    first.emit(trade(amount=3))
    cursor, events, reset = first.read_after()
    assert reset and len(events) == 1
    assert first.read_after(cursor)[1:] == ([], False)
    first.emit(trade(amount=4))
    assert first.read_after(cursor)[2]
    second = EventLog()
    second.emit(trade(amount=5))
    assert second.read_after(cursor)[2]
    current = second.read_after()[0]
    for offset in (-1, 2, True, 1.0):
        with pytest.raises(ValueError, match='cursor'):
            second.read_after((current[0], offset))
