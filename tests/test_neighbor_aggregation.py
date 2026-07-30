"""Relation-graph neighborhood queries — the network-dynamics primitive.

Before these, "infect me if a graph NEIGHBOR is infected" was
inexpressible: relations were readable only as an actor->target boolean.
"""
from fg_env_kernel.pipeline.loader import load_world


def _epidemic_world():
    return load_world({
        "name": "contact net",
        "entity_types": [{
            "name": "Person", "role": "agent",
            "properties": [
                {"name": "infected", "type": "bool", "default": False},
                {"name": "viral_load", "type": "float", "default": 0.0},
            ],
        }],
        "entities": [
            {"id": f"p{i}", "entity_type": "Person", "name": f"P{i}",
             "properties": ({"infected": True, "viral_load": 2.0}
                            if i == 0 else {})}
            for i in range(6)
        ],
        # Ring: p0-p1-p2-p3-p4-p0; p5 is ISOLATED; p0 starts infected.
        "initial_relations": [
            {"from": f"p{i}", "to": f"p{(i + 1) % 5}",
             "relation": "contact", "value": 1.0}
            for i in range(5)
        ],
        "derived_rules": [{
            "name": "contagion",
            "for_each": "$entities_of(Person)",
            "when": "$neighbor_count($params.it, 'contact', 'infected') >= 1",
            "then": [{"target": "actor", "operation": "set",
                      "field": "infected", "value": True}],
            "once_per_entity": True,
        }],
        "temporal": {"max_rounds": 6},
    })


class TestNeighborQueries:
    def test_neighbors_are_bidirectional(self):
        state, engine = _epidemic_world()
        from fg_env_kernel.effects import resolve_expression
        out = resolve_expression("$neighbors('p0', 'contact')", state=state)
        assert out == ["p1", "p4"]

    def test_neighbor_count_filters_on_property(self):
        state, engine = _epidemic_world()
        from fg_env_kernel.effects import resolve_expression
        assert resolve_expression(
            "$neighbor_count('p1', 'contact', 'infected')", state=state) == 1
        assert resolve_expression(
            "$neighbor_count('p2', 'contact', 'infected')", state=state) == 0

    def test_neighbor_sum(self):
        state, engine = _epidemic_world()
        from fg_env_kernel.effects import resolve_expression
        assert resolve_expression(
            "$neighbor_sum('p1', 'contact', 'viral_load')", state=state) == 2.0


class TestDeclarativeContagion:
    def test_infection_spreads_along_the_ring(self):
        state, engine = _epidemic_world()

        def infected():
            return {e.id for e in state.entities.values()
                    if e.properties.get("infected")}

        assert infected() == {"p0"}
        # In-tick updates are sequential (asynchronous-update semantics),
        # so the connected component can cascade within a tick — but
        # spread is strictly ALONG EDGES: the isolated node never
        # catches it, no matter how many rounds pass.
        for _ in range(4):
            state._derived_rules.tick(engine)
        assert infected() == {"p0", "p1", "p2", "p3", "p4"}
        assert "p5" not in infected()
