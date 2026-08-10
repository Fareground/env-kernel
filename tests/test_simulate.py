"""simulate() one-shot API, random_policy, template coercion, World readability."""
import json

import pytest

from fg_env_kernel import ActionInstance, Kernel, random_policy, simulate


def make_template():
    return {
        "name": "Race to 10",
        "description": "Two runners sprint; first to distance 10 wins.",
        "entity_types": [
            {"name": "runner", "role": "agent", "properties": [
                {"name": "distance", "type": "float", "default": 0}
            ]}
        ],
        "entities": [
            {"id": "alice", "entity_type": "runner", "name": "Alice"},
            {"id": "bob", "entity_type": "runner", "name": "Bob"},
        ],
        "actions": [
            {"name": "sprint", "description": "Run forward.", "actor_type": "runner",
             "effects_on_success": [
                 {"operation": "add", "target": "actor", "field": "distance",
                  "value": "$random(1, 3)"}
             ]},
            {"name": "rest", "description": "Catch a breath.", "actor_type": "runner"},
        ],
        "termination_conditions": [
            {"name": "finish_line", "check_type": "expr",
             "params": {"expr": "$state.entities.alice.distance >= 10 || "
                                "$state.entities.bob.distance >= 10"}}
        ],
        "temporal": {"max_rounds": 30},
    }


class TestSimulateZeroConfig:
    def test_runs_to_completion(self):
        world = simulate(make_template())
        assert world.finished
        assert world.current_round >= 1
        assert world.events  # something happened

    def test_returns_finished_world_with_readable_fields(self):
        world = simulate(make_template(), seed=3)
        assert world.terminated_by in ("finish_line", None)
        assert world.seed == 3

    def test_explicit_agent_is_used(self):
        calls = []

        def agent(entity_id, perception, valid_actions):
            calls.append(entity_id)
            return ActionInstance(action_name="sprint", actor_id=entity_id)

        world = simulate(make_template(), agent=agent, seed=1)
        assert world.finished
        assert calls  # our agent drove the run

    def test_max_rounds_override(self):
        world = simulate(make_template(), max_rounds=2, seed=1)
        assert world.current_round <= 2

    def test_on_event_stream(self):
        seen = []
        simulate(make_template(), seed=1, on_event=lambda e: seen.append(e))
        assert seen


class TestDeterminism:
    def test_same_seed_identical_event_log(self):
        a = simulate(make_template(), seed=42)
        b = simulate(make_template(), seed=42)
        assert [(e.event_type, e.narrative) for e in a.events] == \
               [(e.event_type, e.narrative) for e in b.events]
        assert a.terminated_by == b.terminated_by
        assert a.current_round == b.current_round

    def test_different_seeds_allowed_to_diverge(self):
        # Not asserting divergence (tiny action space), just that both finish.
        assert simulate(make_template(), seed=1).finished
        assert simulate(make_template(), seed=2).finished

    def test_default_policy_never_touches_global_random(self):
        import random
        random.seed(123)
        before = random.random()
        random.seed(123)
        simulate(make_template(), seed=42)
        after = random.random()
        assert before == after


class TestRandomPolicy:
    def test_respects_valid_actions(self):
        policy = random_policy(seed=0)
        for _ in range(20):
            action = policy("alice", {}, ["sprint", "rest"])
            assert action is not None
            assert action.action_name in ("sprint", "rest")
            assert action.actor_id == "alice"

    def test_no_valid_actions_skips_turn(self):
        policy = random_policy(seed=0)
        assert policy("alice", {}, []) is None

    def test_deterministic_given_seed(self):
        picks_a = [random_policy(seed=5)("x", {}, ["a", "b", "c"]).action_name
                   for _ in range(1)]
        picks_b = [random_policy(seed=5)("x", {}, ["a", "b", "c"]).action_name
                   for _ in range(1)]
        assert picks_a == picks_b

    def test_state_bound_policy_skips_required_param_actions(self):
        template = make_template()
        template["actions"][0]["parameters"] = [
            {"name": "effort", "type": "int", "required": True}
        ]
        world = Kernel(seed=0).load(template)
        policy = random_policy(seed=0, state=world.state)
        for _ in range(20):
            action = policy("alice", {}, ["sprint", "rest"])
            assert action.action_name == "rest"

    def test_state_bound_policy_picks_valid_target(self):
        template = make_template()
        template["actions"].append(
            {"name": "shove", "description": "Shove a rival.",
             "actor_type": "runner", "target_type": "runner"}
        )
        world = Kernel(seed=0).load(template)
        policy = random_policy(seed=0, state=world.state)
        for _ in range(30):
            action = policy("alice", {}, ["shove"])
            assert action.action_name == "shove"
            assert action.target_id == "bob"  # only other runner


class TestTemplateCoercion:
    def test_path_and_str_loading(self, tmp_path):
        f = tmp_path / "race.json"
        f.write_text(json.dumps(make_template()))
        assert simulate(f, seed=1).finished          # Path
        assert simulate(str(f), seed=1).finished     # str

    def test_missing_file_is_friendly(self):
        with pytest.raises(FileNotFoundError, match="Template file not found"):
            simulate("/nope/does_not_exist.json")

    def test_invalid_json_is_friendly(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            simulate(f)

    def test_non_object_json_is_friendly(self, tmp_path):
        f = tmp_path / "list.json"
        f.write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            simulate(f)

    def test_kernel_load_accepts_path(self, tmp_path):
        f = tmp_path / "race.json"
        f.write_text(json.dumps(make_template()))
        world = Kernel(seed=1).load(f)
        assert not world.finished


class TestWorldReadability:
    def test_repr(self):
        world = simulate(make_template(), seed=42)
        r = repr(world)
        assert r.startswith("<World 'Race to 10'")
        assert "finished" in r
        assert "terminated_by=" in r

    def test_summary(self):
        world = simulate(make_template(), seed=42)
        s = world.summary()
        assert "Race to 10" in s
        assert "finished" in s
        assert "Terminated by:" in s

    def test_unrun_world_reads_as_running(self):
        world = Kernel(seed=1).load(make_template())
        assert "running" in repr(world)
        assert "running" in world.summary()
