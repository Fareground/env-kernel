"""Entry-point template validation — Kernel.load / simulate reject garbage.

Before this gate, ``simulate({"foo": "bar"})`` "succeeded": the template
models use ``extra="allow"``, so a garbage dict validated structurally
and ran 100 empty rounds. Now the pipeline's lint layer (errors only)
runs inside the facade and raises ``TemplateError`` up front.
"""
import logging

import pytest

from fg_env_kernel import Kernel, TemplateError, simulate


def _valid_template():
    return {
        "name": "valid world",
        "entity_types": [{
            "name": "Bidder", "role": "agent",
            "properties": [{"name": "offer", "type": "float", "default": 0.0}],
        }],
        "entities": [{"id": "b1", "entity_type": "Bidder", "name": "B1"}],
        "actions": [{
            "name": "bid",
            "description": "Place a bid",
            "actor_type": "Bidder",
            "effects_on_success": [{"target": "actor", "operation": "set",
                                    "field": "offer", "value": 1.0}],
        }],
        "temporal": {"max_rounds": 2},
    }


class TestGarbageTemplatesRaise:
    def test_garbage_dict_raises_with_helpful_message(self):
        with pytest.raises(TemplateError) as exc:
            simulate({"foo": "bar"})
        msg = str(exc.value)
        assert "role='agent'" in msg          # the blocking error
        assert "foo" in msg                   # the typo'd key is surfaced
        assert exc.value.issues               # structured issues attached

    def test_empty_dict_raises(self):
        with pytest.raises(TemplateError):
            simulate({})

    def test_kernel_load_raises_too(self):
        with pytest.raises(TemplateError):
            Kernel().load({"foo": "bar"})

    def test_unknown_action_operation_is_an_error(self):
        t = _valid_template()
        t["actions"][0]["effects_on_success"][0]["operation"] = "does_not_exist"
        with pytest.raises(TemplateError, match="does_not_exist"):
            Kernel().load(t)


class TestValidTemplatesUnaffected:
    def test_valid_template_loads(self):
        world = Kernel(seed=1).load(_valid_template())
        assert not world.finished

    def test_valid_template_simulates(self):
        world = simulate(_valid_template(), seed=1)
        assert world.finished


class TestWarnings:
    def test_warnings_do_not_raise_by_default(self, caplog):
        t = _valid_template()
        # no termination_conditions → lint WARNING, not an error
        with caplog.at_level(logging.WARNING, logger="fg_env_kernel.kernel"):
            world = Kernel().load(t)
        assert not world.finished
        assert any("termination_conditions" in r.message
                   for r in caplog.records)

    def test_strict_raises_on_warnings(self):
        with pytest.raises(TemplateError):
            Kernel().load(_valid_template(), strict=True)

    def test_strict_ok_on_clean_template(self):
        t = _valid_template()
        t["termination_conditions"] = [{
            "check_type": "score_after_n_rounds", "params": {"rounds": 2},
        }]
        world = Kernel().load(t, strict=True)
        assert not world.finished


class TestCustomRegistryNoFalsePositive:
    def test_forked_registry_effect_passes_lint(self):
        from fg_env_kernel import registry

        mine = registry.fork()

        @mine.effect("my_custom_op")
        def _my_op(ctx, spec):  # pragma: no cover - never invoked here
            return None

        t = _valid_template()
        t["actions"][0]["effects_on_success"][0] = {
            "target": "actor", "operation": "my_custom_op",
        }
        world = Kernel(registry=mine).load(t)
        assert not world.finished
