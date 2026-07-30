"""Predicate comparisons fail CLOSED on malformed input.

`actor.trust > 99.0` (missing $) used to evaluate True via lexicographic
string comparison — silently inverting the author's declared gate.
"""
from fg_env_kernel.predicates import evaluate


class _Ent:
    def __init__(self, **props):
        self.properties = dict(props)

    def get(self, k):
        return self.properties.get(k)


class TestFailClosed:
    def test_bare_dotted_identifier_never_fires(self):
        actor = _Ent(trust=0.5)
        assert evaluate("actor.trust > 99.0", actor=actor) is False
        assert evaluate("actor.trust < 99.0", actor=actor) is False

    def test_the_dollar_form_still_works(self):
        actor = _Ent(trust=0.5)
        assert evaluate("$actor.trust < 99.0", actor=actor) is True
        assert evaluate("$actor.trust > 99.0", actor=actor) is False

    def test_numeric_vs_string_comparison_is_false(self):
        actor = _Ent(status="active")
        assert evaluate("$actor.status > 5", actor=actor) is False
        assert evaluate("$actor.status < 5", actor=actor) is False

    def test_bare_enum_literals_still_compare(self):
        actor = _Ent(status="active")
        assert evaluate("$actor.status == active", actor=actor) is True

    def test_string_ordering_still_works_between_strings(self):
        actor = _Ent(tier="bronze")
        assert evaluate("$actor.tier < 'silver'", actor=actor) is True


class TestScientificNotation:
    def test_tiny_probabilities_do_not_become_certainties(self):
        """`1e-06` tokenized as `1` — a one-in-a-million roll fired with
        certainty."""
        hits = sum(
            1 for _ in range(200)
            if evaluate("$random_float(0, 1) < 1e-06")
        )
        assert hits == 0

    def test_exponent_forms_parse(self):
        assert evaluate("2e3 > 1999") is True
        assert evaluate("1.5E-2 < 0.02") is True
        assert evaluate("1e+2 == 100") is True


class TestPowerOperator:
    def test_caret_and_double_star_mean_power(self):
        """`^` used to evaluate as truthiness (fail-open gate inversion);
        `**` returned None (permanently-False gate). Both now compute."""
        actor = _Ent(x=0.4)
        assert evaluate("$actor.x ^ 2 > 1", actor=actor) is False
        assert evaluate("$actor.x ** 2 < 1", actor=actor) is True
        actor2 = _Ent(x=3.0)
        assert evaluate("$actor.x ^ 2 > 8", actor=actor2) is True
        assert evaluate("$actor.x ** 2 == 9", actor=actor2) is True
