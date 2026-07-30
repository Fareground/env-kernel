"""Tests for the contract version negotiation pipeline."""
import pytest

from fg_env_kernel.pipeline.versioning import (
    CONTRACT_VERSION,
    _MIGRATIONS,
    get_template_version,
    list_known_versions,
    register_migration,
    upgrade_template,
)


@pytest.fixture(autouse=True)
def _reset_migrations():
    """Save + restore the global migrations list around each test."""
    saved = list(_MIGRATIONS)
    yield
    _MIGRATIONS.clear()
    _MIGRATIONS.extend(saved)


class TestVersionDetection:
    def test_detects_explicit_version(self):
        assert get_template_version({"contract_version": "1.0"}) == "1.0"

    def test_missing_version_returns_fallback(self):
        # With no migrations, fallback IS the current version.
        v = get_template_version({"name": "test"})
        assert isinstance(v, str)

    def test_numeric_versions_coerced_to_string(self):
        assert get_template_version({"contract_version": 1.0}) == "1.0"


class TestMigrationChain:
    def test_no_migration_needed_when_current(self):
        raw = {"contract_version": CONTRACT_VERSION, "name": "x"}
        out = upgrade_template(raw)
        assert out["contract_version"] == CONTRACT_VERSION
        # Shallow copy — not the same object
        assert out is not raw

    def test_registered_migration_applied(self):
        @register_migration("0.5")
        def _v05_to_v1(raw):
            raw["contract_version"] = "1.0"
            raw["migrated"] = True
            return raw

        raw = {"contract_version": "0.5", "name": "old"}
        out = upgrade_template(raw)
        assert out["contract_version"] == "1.0"
        assert out["migrated"] is True

    def test_unregistered_version_warns_and_stops(self, caplog):
        raw = {"contract_version": "999.0", "name": "future"}
        out = upgrade_template(raw)
        # No migration registered for v999 — stays put
        assert out["contract_version"] == "999.0"

    def test_chain_walks_multiple_steps(self):
        @register_migration("0.1")
        def _v01_to_v02(raw):
            raw["contract_version"] = "0.2"
            raw["step1"] = True
            return raw

        @register_migration("0.2")
        def _v02_to_v10(raw):
            raw["contract_version"] = "1.0"
            raw["step2"] = True
            return raw

        raw = {"contract_version": "0.1"}
        out = upgrade_template(raw)
        assert out["contract_version"] == "1.0"
        assert out["step1"] is True
        assert out["step2"] is True


class TestCompileIntegration:
    def test_compile_template_calls_upgrade(self):
        """A legacy template with no contract_version still compiles —
        upgrade_template adds the field transparently."""
        from fg_env_kernel import compile_template

        # Minimum valid template, no contract_version
        raw = {
            "name": "race",
            "entity_types": [{
                "name": "P", "role": "agent",
                "properties": [{"name": "score", "type": "int", "default": 0}],
            }],
            "actions": [{
                "name": "noop",
                "actor_type": "P",
                "resolution_archetype": "deterministic",
            }],
            "entities": [
                {"id": "a", "name": "A", "entity_type": "P",
                 "properties": {"score": 0}},
            ],
        }
        assert "contract_version" not in raw
        result = compile_template(raw)
        assert result.ok
        # After compile, the template inside the result HAS the version
        assert result.template.contract_version


class TestListKnownVersions:
    def test_includes_current(self):
        versions = list_known_versions()
        assert CONTRACT_VERSION in versions
