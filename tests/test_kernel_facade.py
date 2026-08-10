"""SDK facade — Kernel/World wrap the (WorldState, SimulationEngine) pair."""
import json
from pathlib import Path

import pytest

from fg_env_kernel import (
    ActionInstance,
    DecisionFn,
    Kernel,
    SimulationEngine,
    World,
    WorldState,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


def _template():
    return {
        "name": "facade world",
        "entity_types": [{
            "name": "Bidder", "role": "agent",
            "properties": [{"name": "offer", "type": "float", "default": 0.0}],
        }],
        "entities": [{"id": "b1", "entity_type": "Bidder", "name": "B1"}],
        "actions": [{
            "name": "bid",
            "description": "Place a bid",
            "actor_type": "Bidder",
            "parameters": [{"name": "amount", "type": "float",
                            "min": 1.0, "max": 100.0}],
            "effects_on_success": [{"target": "actor", "operation": "set",
                                    "field": "offer",
                                    "value": "$params.amount"}],
        }],
        "temporal": {"max_rounds": 3},
    }


def _bid(amount: float) -> DecisionFn:
    def decision_fn(entity_id, perception, valid_actions):
        assert isinstance(perception, dict)
        assert "bid" in valid_actions
        return ActionInstance(action_name="bid", actor_id=entity_id,
                              parameters={"amount": amount})
    return decision_fn


class TestKernelLoad:
    def test_load_returns_world_wrapping_state_and_engine(self):
        world = Kernel(seed=42).load(_template())
        assert isinstance(world, World)
        assert isinstance(world.state, WorldState)
        assert isinstance(world.engine, SimulationEngine)
        assert world.seed == 42
        assert not world.finished

    def test_template_max_rounds_reaches_engine(self):
        world = Kernel().load(_template())
        assert world.engine.max_rounds == 3

    def test_explicit_max_rounds_overrides_template(self):
        world = Kernel().load(_template(), max_rounds=5)
        assert world.engine.max_rounds == 5

    def test_load_seed_overrides_kernel_seed(self):
        world = Kernel(seed=1).load(_template(), seed=99)
        assert world.seed == 99


class TestRun:
    def test_run_to_completion(self):
        world = Kernel(seed=42).load(_template(), decision_fn=_bid(42.5))
        world.run()
        assert world.finished
        assert world.current_round == 3
        assert world.state.get_entity("b1").properties["offer"] == 42.5
        types = [e.event_type for e in world.events]
        assert types[0] == "simulation_start"
        assert types[-1] == "simulation_end"

    def test_on_event_streams(self):
        seen = []
        world = Kernel().load(_template(), decision_fn=_bid(2.0),
                              on_event=lambda e: seen.append(e))
        world.run()
        assert seen and len(seen) == len(world.events)


class TestStep:
    def test_stepwise_matches_round_count(self):
        world = Kernel(seed=42).load(_template(), decision_fn=_bid(10.0))
        world.step()
        assert world.current_round == 1
        assert not world.finished
        world.step()
        world.step()
        assert world.finished
        types = [e.event_type for e in world.events]
        assert types[0] == "simulation_start"
        assert types[-1] == "simulation_end"
        assert types.count("simulation_start") == 1
        assert types.count("simulation_end") == 1

    def test_step_after_finish_is_noop(self):
        world = Kernel().load(_template(), decision_fn=_bid(10.0))
        world.run()
        n_events = len(world.events)
        world.step()
        assert len(world.events) == n_events
        assert world.current_round == 3

    def test_step_matches_run_result(self):
        stepped = Kernel(seed=7).load(_template(), decision_fn=_bid(33.0))
        while not stepped.finished:
            stepped.step()
        ran = Kernel(seed=7).load(_template(), decision_fn=_bid(33.0))
        ran.run()
        assert (stepped.state.get_entity("b1").properties["offer"]
                == ran.state.get_entity("b1").properties["offer"])
        assert stepped.current_round == ran.current_round

    def test_step_rejects_continuous_mode(self):
        template = _template()
        template["temporal"] = {
            "mode": "continuous", "max_rounds": 3,
            "continuous": {"max_time": 3.0},
        }
        world = Kernel().load(_template() | {"temporal": template["temporal"]})
        with pytest.raises(RuntimeError, match="discrete"):
            world.step()


class TestExamples:
    def test_tic_tac_toe_template_loads_and_terminates(self):
        template = json.loads(
            (EXAMPLES / "tic_tac_toe" / "template.json").read_text())

        def decision_fn(entity_id, perception, valid_actions):
            # Always aim at cell 1; the engine falls back to the first
            # empty cell when it's taken, so the board steadily fills.
            return ActionInstance(action_name="place_mark",
                                  actor_id=entity_id,
                                  parameters={"cell": 1})

        world = Kernel(seed=3).load(template, decision_fn=decision_fn)
        world.run()
        assert world.finished
        assert world.terminated_by == "three_in_a_row"
