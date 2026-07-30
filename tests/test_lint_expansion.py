"""Lint expansion tests — catch the failure modes a real env-builder
agent hit:

  - invalid effect targets like `target_owner` (silently dropped at runtime)
  - unquoted bareword comparisons in expressions: `== neutral`
  - $-refs that don't resolve: `$actor.entity_id` (should be `$actor.id`)
"""
import pytest

from fg_env_kernel import lint_template


BASE = {
    "name": "lint_test",
    "entity_types": [{
        "name": "P", "role": "agent",
        "properties": [
            {"name": "gold", "type": "int", "default": 0},
            {"name": "color", "type": "string", "default": "red"},
        ],
    }],
    "entities": [
        {"id": "alice", "name": "A", "entity_type": "P",
         "properties": {"gold": 10, "color": "red"}},
    ],
    "termination_conditions": [{
        "name": "stop", "check_type": "round_limit", "params": {"max_rounds": 10}
    }],
}


def _issue_paths(issues):
    return {i.path for i in issues}


def _messages(issues):
    return [i.message for i in issues]


class TestInvalidEffectTargets:
    def test_made_up_target_is_caught(self):
        bad = {**BASE}
        bad["actions"] = [{
            "name": "claim",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "set", "target": "target_owner",  # invalid!
                 "field": "color", "value": "blue"}
            ],
        }]
        issues = lint_template(bad)
        errs = [i for i in issues if i.severity == "error"]
        assert any("target_owner" in i.message for i in errs)

    def test_actor_target_string_is_ok(self):
        ok = {**BASE}
        ok["actions"] = [{
            "name": "earn",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "add", "target": "actor",
                 "field": "gold", "value": 1}
            ],
        }]
        issues = lint_template(ok)
        # No target-related errors
        assert not any("target" in i.message and "not a recognized" in i.message
                       for i in issues)

    def test_dollar_expression_targets_pass(self):
        ok = {**BASE}
        ok["actions"] = [{
            "name": "give",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "add", "target": "$params.target_id",
                 "field": "gold", "value": 5}
            ],
        }]
        issues = lint_template(ok)
        # $-expressions are unverifiable statically — should not warn/error here
        assert not any("not a recognized" in i.message for i in issues)

    def test_pool_target_is_ok(self):
        ok = {**BASE}
        ok["resource_types"] = [{"name": "treasury"}]
        ok["actions"] = [{
            "name": "tax",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "transfer_resource", "target": "pool:treasury",
                 "resource": "treasury", "value": 1}
            ],
        }]
        issues = lint_template(ok)
        assert not any("not a recognized" in i.message for i in issues)


class TestBarewordExpressions:
    def test_unquoted_bareword_caught(self):
        bad = {**BASE}
        bad["actions"] = [{
            "name": "go",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "preconditions": [
                # Compares to bareword 'neutral' — should be quoted
                {"expr": "$actor.color == neutral"}
            ],
            "effects_on_success": [],
        }]
        issues = lint_template(bad)
        warns = [i for i in issues if i.severity == "warning"]
        assert any("neutral" in w.message and "bareword" in w.message
                   for w in warns)

    def test_quoted_string_ok(self):
        ok = {**BASE}
        ok["actions"] = [{
            "name": "go",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "preconditions": [
                {"expr": "$actor.color == 'neutral'"}
            ],
            "effects_on_success": [],
        }]
        issues = lint_template(ok)
        # No bareword warning
        assert not any("bareword" in i.message for i in issues)

    def test_reserved_keywords_not_flagged(self):
        ok = {**BASE}
        ok["actions"] = [{
            "name": "g",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "preconditions": [
                {"expr": "$actor.alive == true && $actor.gold != null"}
            ],
            "effects_on_success": [],
        }]
        issues = lint_template(ok)
        assert not any("bareword" in i.message for i in issues)


class TestProblematicRefs:
    def test_actor_entity_id_caught(self):
        bad = {**BASE}
        bad["actions"] = [{
            "name": "tag",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                # Should be $actor.id
                {"operation": "set", "target": "actor",
                 "field": "tag", "value": "$actor.entity_id"}
            ],
        }]
        issues = lint_template(bad)
        warns = [i for i in issues if i.severity == "warning"]
        assert any("entity_id" in w.message for w in warns)

    def test_correct_id_ref_ok(self):
        ok = {**BASE}
        ok["actions"] = [{
            "name": "tag",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "effects_on_success": [
                {"operation": "set", "target": "actor",
                 "field": "tag", "value": "$actor.id"}
            ],
        }]
        issues = lint_template(ok)
        assert not any("entity_id" in i.message for i in issues)


def test_lint_runs_on_clean_template_with_zero_issues():
    """The acceptance gate: a real schema with rules / derived_rules /
    composition primitives should lint clean."""
    clean = {
        **BASE,
        "actions": [{
            "name": "earn",
            "actor_type": "P",
            "resolution_archetype": "deterministic",
            "preconditions": [{"expr": "$actor.alive && $actor.gold < 100"}],
            "effects_on_success": [
                {"operation": "add", "target": "actor", "field": "gold", "value": 1}
            ],
        }],
        "derived_rules": [{
            "name": "broke",
            "when": "$params.it.gold <= 0",
            "for_each": "$entities_of(P)",
            "as": "it",
            "then": [{"operation": "set", "target": "$params.it",
                      "field": "color", "value": "'red'"}],
            "once_per_entity": True,
        }],
    }
    issues = lint_template(clean)
    errs = [i for i in issues if i.severity == "error"]
    assert errs == [], f"expected no errors, got: {[str(e) for e in errs]}"
