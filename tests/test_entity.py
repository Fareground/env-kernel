"""Tests for kernel/entity.py"""
import pytest
from fg_env_kernel.entity import EntityType, Entity
from fg_env_kernel.types import PropertySchema, PropertyType


class TestEntityType:
    def test_create_entity_type(self):
        et = EntityType(
            name="warrior",
            role="agent",
            properties=[
                PropertySchema(name="health", type=PropertyType.FLOAT, default=100.0, min_value=0, max_value=100),
                PropertySchema(name="strength", type=PropertyType.INT, default=10),
            ],
        )
        assert et.name == "warrior"
        assert et.role == "agent"
        assert len(et.properties) == 2

    def test_get_property_schema(self):
        et = EntityType(
            name="warrior",
            role="agent",
            properties=[
                PropertySchema(name="health", type=PropertyType.FLOAT),
                PropertySchema(name="strength", type=PropertyType.INT),
            ],
        )
        assert et.get_property_schema("health") is not None
        assert et.get_property_schema("health").type == PropertyType.FLOAT
        assert et.get_property_schema("nonexistent") is None


class TestEntity:
    def test_create_entity(self):
        e = Entity(id="e1", name="Warrior Bob", entity_type="warrior", properties={"health": 100.0, "strength": 15})
        assert e.id == "e1"
        assert e.name == "Warrior Bob"
        assert e.alive is True

    def test_get_set(self):
        e = Entity(id="e1", name="Bob", entity_type="warrior", properties={"health": 100.0})
        assert e.get("health") == 100.0
        assert e.get("nonexistent", 0) == 0
        e.set("health", 50.0)
        assert e.get("health") == 50.0

    def test_modify_without_schema(self):
        e = Entity(id="e1", name="Bob", entity_type="warrior", properties={"health": 100.0})
        e.modify("health", -25.0)
        assert e.get("health") == 75.0

    def test_modify_with_schema_bounds(self):
        schema = PropertySchema(name="health", type=PropertyType.FLOAT, min_value=0, max_value=100)
        e = Entity(id="e1", name="Bob", entity_type="warrior", properties={"health": 10.0})
        e.modify("health", -50.0, schema)
        assert e.get("health") == 0.0  # Clamped to min

        e.set("health", 90.0)
        e.modify("health", 50.0, schema)
        assert e.get("health") == 100.0  # Clamped to max

    def test_modify_non_numeric_raises(self):
        e = Entity(id="e1", name="Bob", entity_type="warrior", properties={"name": "Bob"})
        with pytest.raises(ValueError):
            e.modify("name", 5.0)

    def test_to_dict(self):
        e = Entity(id="e1", name="Bob", entity_type="warrior", properties={"health": 100.0}, location_id="town")
        d = e.to_dict()
        assert d["id"] == "e1"
        assert d["name"] == "Bob"
        assert d["properties"]["health"] == 100.0
        assert d["location_id"] == "town"
        assert d["alive"] is True
