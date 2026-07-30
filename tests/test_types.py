"""Tests for kernel/types.py"""
import pytest
from fg_env_kernel.types import PropertyType, PropertySchema


class TestPropertySchema:
    def test_validate_float(self):
        schema = PropertySchema(name="health", type=PropertyType.FLOAT, min_value=0, max_value=100)
        assert schema.validate(50.0) is True
        assert schema.validate(0.0) is True
        assert schema.validate(100.0) is True
        assert schema.validate(-1.0) is False
        assert schema.validate(101.0) is False
        assert schema.validate("not a float") is False
        assert schema.validate(None) is True  # None is always valid

    def test_validate_int(self):
        schema = PropertySchema(name="level", type=PropertyType.INT, min_value=1, max_value=10)
        assert schema.validate(5) is True
        assert schema.validate(1) is True
        assert schema.validate(0) is False
        assert schema.validate(11) is False
        assert schema.validate(5.5) is False  # Floats invalid for INT

    def test_validate_string(self):
        schema = PropertySchema(name="name", type=PropertyType.STRING)
        assert schema.validate("hello") is True
        assert schema.validate(42) is False

    def test_validate_bool(self):
        schema = PropertySchema(name="active", type=PropertyType.BOOL)
        assert schema.validate(True) is True
        assert schema.validate(False) is True
        assert schema.validate(1) is False

    def test_validate_enum(self):
        schema = PropertySchema(name="role", type=PropertyType.ENUM, enum_values=["warrior", "mage", "rogue"])
        assert schema.validate("warrior") is True
        assert schema.validate("bard") is False

    def test_validate_list(self):
        schema = PropertySchema(name="items", type=PropertyType.LIST)
        assert schema.validate(["sword", "shield"]) is True
        assert schema.validate("not a list") is False

    def test_coerce_float(self):
        schema = PropertySchema(name="x", type=PropertyType.FLOAT)
        assert schema.coerce(42) == 42.0
        assert schema.coerce("3.14") == 3.14

    def test_coerce_int(self):
        schema = PropertySchema(name="x", type=PropertyType.INT)
        assert schema.coerce(3.7) == 3
        assert schema.coerce("5") == 5

    def test_coerce_none_returns_default(self):
        schema = PropertySchema(name="x", type=PropertyType.FLOAT, default=10.0)
        assert schema.coerce(None) == 10.0

    def test_coerce_enum_invalid(self):
        schema = PropertySchema(name="role", type=PropertyType.ENUM, enum_values=["a", "b"])
        with pytest.raises(ValueError):
            schema.coerce("c")

    def test_clamp(self):
        schema = PropertySchema(name="hp", type=PropertyType.FLOAT, min_value=0, max_value=100)
        assert schema.clamp(150) == 100
        assert schema.clamp(-10) == 0
        assert schema.clamp(50) == 50

    def test_clamp_int(self):
        schema = PropertySchema(name="level", type=PropertyType.INT, min_value=1, max_value=10)
        assert schema.clamp(15) == 10
        assert isinstance(schema.clamp(5), int)
