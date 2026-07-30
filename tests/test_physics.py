"""Tests for the coupled-dynamics physics integrator.

These verify the integrator against KNOWN analytic solutions and conservation
laws — the right way to test an ODE solver: it's not "does it run" but "does it
match the math".
"""
import math

import pytest

from fg_env_kernel.physics import (
    EntitySource,
    EntityWriteback,
    PhysicsExprError,
    PhysicsModel,
    PhysicsVariable,
)


# ---------------------------------------------------------------------------
# Safe evaluator
# ---------------------------------------------------------------------------

def test_rejects_attribute_access():
    with pytest.raises(PhysicsExprError):
        PhysicsModel(variables=[PhysicsVariable("x", 1.0, rate="x.__class__")])


def test_rejects_unknown_symbol():
    with pytest.raises(PhysicsExprError):
        PhysicsModel(variables=[PhysicsVariable("x", 1.0, rate="x * mystery_param")])


def test_rejects_call_to_non_whitelisted():
    with pytest.raises(PhysicsExprError):
        PhysicsModel(variables=[PhysicsVariable("x", 1.0, rate="__import__('os')")])


def test_comparison_collapses_to_numeric():
    # (x > 0) gates the decay term: positive x decays, non-positive holds.
    m = PhysicsModel(variables=[PhysicsVariable("x", 5.0, rate="-(x > 0) * x")])
    m.integrate(1.0)
    assert m.values["x"] < 5.0


# ---------------------------------------------------------------------------
# Analytic correctness
# ---------------------------------------------------------------------------

def test_exponential_decay_matches_analytic():
    # dx/dt = -k x  =>  x(t) = x0 * e^(-k t)
    k = 0.5
    x0 = 100.0
    m = PhysicsModel(
        variables=[PhysicsVariable("x", x0, rate="-k * x")],
        params={"k": k}, substeps=16,
    )
    T = 4.0
    m.integrate(T)
    expected = x0 * math.exp(-k * T)
    assert m.values["x"] == pytest.approx(expected, rel=1e-4)
    assert m.time == pytest.approx(T)


def test_logistic_saturates_to_carrying_capacity():
    # dx/dt = r x (1 - x/K)  ->  x -> K
    m = PhysicsModel(
        variables=[PhysicsVariable("x", 1.0, rate="r * x * (1 - x / K)", min=0.0)],
        params={"r": 1.0, "K": 1000.0}, substeps=8,
    )
    for _ in range(40):
        m.integrate(1.0)
    assert m.values["x"] == pytest.approx(1000.0, rel=1e-3)


def test_dt_invariance_one_big_vs_many_small():
    # Integrating dt=4 once should match dt=1 four times (within RK4 error).
    def decay():
        return PhysicsModel(
            variables=[PhysicsVariable("x", 50.0, rate="-0.3 * x")], substeps=8
        )
    big = decay(); big.integrate(4.0)
    small = decay()
    for _ in range(4):
        small.integrate(1.0)
    assert big.values["x"] == pytest.approx(small.values["x"], rel=1e-4)


def test_time_dependent_rate_matches_analytic():
    # dx/dt = cos(t), x(0)=0  =>  x(t) = sin(t).
    # This FAILS if RK4 evaluates intermediate stages at the wrong time.
    m = PhysicsModel(variables=[PhysicsVariable("x", 0.0, rate="cos(t)")], substeps=16)
    m.integrate(math.pi / 2)          # x should reach sin(pi/2) = 1
    assert m.values["x"] == pytest.approx(1.0, abs=1e-4)
    m.integrate(math.pi / 2)          # to t=pi: x = sin(pi) = 0
    assert m.values["x"] == pytest.approx(0.0, abs=1e-4)


def test_reject_min_greater_than_max():
    with pytest.raises(PhysicsExprError):
        PhysicsModel(variables=[PhysicsVariable("x", 5.0, rate="-x", min=100.0, max=10.0)])


def test_reject_source_and_rate_conflict():
    with pytest.raises(PhysicsExprError):
        PhysicsModel(variables=[
            PhysicsVariable("x", 0.0, rate="-x", source=EntitySource("e", "p", "sum")),
        ])


def test_reject_non_numeric_constant():
    with pytest.raises(PhysicsExprError):
        PhysicsModel(variables=[PhysicsVariable("x", 1.0, rate="x * 'boom'")])


def test_division_by_zero_skips_gracefully():
    # Rate 1/x with x driven to 0 must not crash the model.
    m = PhysicsModel(variables=[PhysicsVariable("x", 0.0, rate="1 / x")])
    changes = m.integrate(1.0)
    assert any(c["type"] == "physics_error" for c in changes)
    assert m.values["x"] == 0.0  # value preserved, no crash


def test_overflow_skips_gracefully():
    # exp of a fast-growing value overflows; must degrade, not raise.
    m = PhysicsModel(variables=[PhysicsVariable("x", 700.0, rate="exp(x)")], substeps=1)
    changes = m.integrate(10.0)
    assert any(c["type"] == "physics_error" for c in changes)


def test_no_ode_path_still_writes_back():
    # A pure source->writeback passthrough (no rate) still syncs entities.
    src = [_FakeEntity("a", "sensor", {"reading": 42})]
    dst = [_FakeEntity("b", "display", {"shown": 0})]
    state = _FakeState(src + dst)
    m = PhysicsModel(variables=[
        PhysicsVariable("signal", 0.0,
                        source=EntitySource("sensor", "reading", "sum"),
                        writeback=EntityWriteback("display", "shown", "broadcast")),
    ])
    m.integrate(1.0, state=state)
    assert dst[0].get("shown") == pytest.approx(42.0)


def test_negative_dt_is_noop():
    m = PhysicsModel(variables=[PhysicsVariable("x", 10.0, rate="-x")])
    m.integrate(-1.0)
    assert m.values["x"] == 10.0
    assert m.time == 0.0


# ---------------------------------------------------------------------------
# Coupled systems
# ---------------------------------------------------------------------------

def test_sir_conserves_population():
    # SIR: dS=-bSI/N, dI=bSI/N-gI, dR=gI  =>  S+I+R is invariant.
    N = 1000.0
    m = PhysicsModel(
        variables=[
            PhysicsVariable("S", 990.0, rate="-beta * S * I / N", min=0.0),
            PhysicsVariable("I", 10.0, rate="beta * S * I / N - gamma * I", min=0.0),
            PhysicsVariable("R", 0.0, rate="gamma * I", min=0.0),
        ],
        params={"beta": 0.4, "gamma": 0.1, "N": N}, substeps=16,
    )
    for _ in range(60):
        m.integrate(1.0)
    total = m.values["S"] + m.values["I"] + m.values["R"]
    assert total == pytest.approx(N, rel=1e-4)
    # An epidemic actually happened: most of S moved to R.
    assert m.values["R"] > 500.0
    assert m.values["I"] < m.values["R"]


def test_lotka_volterra_stays_positive_and_oscillates():
    m = PhysicsModel(
        variables=[
            PhysicsVariable("prey", 10.0, rate="alpha*prey - beta*prey*pred", min=0.0),
            PhysicsVariable("pred", 5.0, rate="delta*prey*pred - gamma*pred", min=0.0),
        ],
        params={"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4}, substeps=32,
    )
    prey_series = []
    for _ in range(100):
        m.integrate(0.2)
        prey_series.append(m.values["prey"])
        assert m.values["prey"] >= 0.0
        assert m.values["pred"] >= 0.0
    # Oscillation: the prey population both rose above and fell below its start.
    assert max(prey_series) > 10.0
    assert min(prey_series) < 10.0


def test_clamp_floor_prevents_negative():
    m = PhysicsModel(variables=[PhysicsVariable("x", 1.0, rate="-100 * x", min=0.0)])
    m.integrate(5.0)
    assert m.values["x"] >= 0.0


# ---------------------------------------------------------------------------
# Entity binding (source / writeback)
# ---------------------------------------------------------------------------

class _FakeEntity:
    def __init__(self, eid, etype, props):
        self.id = eid
        self.entity_type = etype
        self.alive = True
        self.properties = dict(props)

    def get(self, k, default=None):
        return self.properties.get(k, default)

    def set(self, k, v):
        self.properties[k] = v


class _FakeState:
    def __init__(self, entities):
        self._entities = entities

    def get_entities_by_type(self, etype):
        return [e for e in self._entities if e.entity_type == etype]


def test_source_reads_aggregate():
    state = _FakeState([
        _FakeEntity("f1", "firm", {"inventory": 30}),
        _FakeEntity("f2", "firm", {"inventory": 70}),
    ])
    m = PhysicsModel(variables=[
        PhysicsVariable("total_inv", 0.0, source=EntitySource("firm", "inventory", "sum")),
    ])
    m.integrate(1.0, state=state)
    assert m.values["total_inv"] == pytest.approx(100.0)


def test_writeback_broadcasts_to_entities():
    traders = [_FakeEntity("t1", "trader", {"px": 0}), _FakeEntity("t2", "trader", {"px": 0})]
    state = _FakeState(traders)
    m = PhysicsModel(variables=[
        PhysicsVariable("price", 100.0, rate="0.0",
                        writeback=EntityWriteback("trader", "px", "broadcast")),
    ])
    m.integrate(1.0, state=state)
    assert traders[0].get("px") == pytest.approx(100.0)
    assert traders[1].get("px") == pytest.approx(100.0)


def test_coupled_source_drives_rate():
    # Price rises with excess demand read from the entity graph.
    buyers = [_FakeEntity("b1", "buyer", {"demand": 8})]
    sellers = [_FakeEntity("s1", "seller", {"supply": 3})]
    state = _FakeState(buyers + sellers)
    m = PhysicsModel(
        variables=[
            PhysicsVariable("demand", 0.0, source=EntitySource("buyer", "demand", "sum")),
            PhysicsVariable("supply", 0.0, source=EntitySource("seller", "supply", "sum")),
            PhysicsVariable("price", 100.0, rate="k * (demand - supply)", min=0.0),
        ],
        params={"k": 0.5}, substeps=4,
    )
    m.integrate(1.0, state=state)
    assert m.values["price"] > 100.0  # excess demand pushed price up


# ---------------------------------------------------------------------------
# Determinism + serialization
# ---------------------------------------------------------------------------

def test_deterministic():
    def run():
        m = PhysicsModel(
            variables=[PhysicsVariable("x", 7.0, rate="sin(t) - 0.1*x")], substeps=8
        )
        for _ in range(20):
            m.integrate(0.3)
        return m.values["x"]
    assert run() == run()


def test_serialization_roundtrip():
    m = PhysicsModel(
        variables=[
            PhysicsVariable("prey", 10.0, rate="a*prey - b*prey*pred", min=0.0,
                            writeback=EntityWriteback("rabbit", "count", "distribute")),
            PhysicsVariable("pred", 5.0, rate="c*prey*pred - d*pred", min=0.0),
        ],
        params={"a": 1.1, "b": 0.4, "c": 0.1, "d": 0.4}, substeps=8,
    )
    m.integrate(2.0)
    restored = PhysicsModel.from_dict(m.to_dict())
    assert restored.values == pytest.approx(m.values)
    assert restored.time == pytest.approx(m.time)
    assert restored.params == m.params
    # And it keeps integrating identically from the restored point.
    m.integrate(1.0)
    restored.integrate(1.0)
    assert restored.values == pytest.approx(m.values)


def test_from_schema_absent_returns_none():
    assert PhysicsModel.from_schema(None) is None
    assert PhysicsModel.from_schema({}) is None
    assert PhysicsModel.from_schema({"params": {"k": 1}}) is None  # no variables


def test_from_schema_builds_model():
    m = PhysicsModel.from_schema({
        "params": {"r": 0.5, "K": 100},
        "variables": [{"name": "pop", "value": 5, "rate": "r*pop*(1-pop/K)", "min": 0}],
        "substeps": 8,
    })
    assert m is not None
    assert "pop" in m.variables
    m.integrate(1.0)
    assert m.values["pop"] > 5.0
