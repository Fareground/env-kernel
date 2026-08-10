"""Tests for kernel/resource.py"""
from fg_env_kernel.resource import ResourceType, ResourcePool


class TestResourcePool:
    def test_basic_operations(self):
        rt = ResourceType(name="gold", conservation=True, discrete=True)
        pool = ResourcePool(resource_type=rt, unallocated=100)

        assert pool.total == 100
        assert pool.get("player1") == 0.0

    def test_add_with_conservation(self):
        rt = ResourceType(name="gold", conservation=True, discrete=True)
        pool = ResourcePool(resource_type=rt, unallocated=100)

        assert pool.add("player1", 50) is True
        assert pool.get("player1") == 50
        assert pool.unallocated == 50
        assert pool.total == 100  # Conservation holds

    def test_add_exceeds_unallocated(self):
        rt = ResourceType(name="gold", conservation=True, discrete=True)
        pool = ResourcePool(resource_type=rt, unallocated=30)

        assert pool.add("player1", 50) is False
        assert pool.get("player1") == 0.0

    def test_add_without_conservation(self):
        rt = ResourceType(name="points", conservation=False, discrete=True)
        pool = ResourcePool(resource_type=rt)

        assert pool.add("player1", 100) is True
        assert pool.get("player1") == 100

    def test_transfer(self):
        rt = ResourceType(name="gold", conservation=True, discrete=True)
        pool = ResourcePool(resource_type=rt, holdings={"p1": 100, "p2": 50})

        assert pool.transfer("p1", "p2", 30) is True
        assert pool.get("p1") == 70
        assert pool.get("p2") == 80

    def test_transfer_insufficient(self):
        rt = ResourceType(name="gold", conservation=True, discrete=True)
        pool = ResourcePool(resource_type=rt, holdings={"p1": 10})

        assert pool.transfer("p1", "p2", 20) is False
        assert pool.get("p1") == 10  # Unchanged

    def test_transfer_zero_or_negative(self):
        rt = ResourceType(name="gold", conservation=True, discrete=True)
        pool = ResourcePool(resource_type=rt, holdings={"p1": 100})

        assert pool.transfer("p1", "p2", 0) is False
        assert pool.transfer("p1", "p2", -5) is False

    def test_remove_with_conservation(self):
        rt = ResourceType(name="gold", conservation=True, discrete=True)
        pool = ResourcePool(resource_type=rt, holdings={"p1": 100}, unallocated=0)

        assert pool.remove("p1", 30) is True
        assert pool.get("p1") == 70
        assert pool.unallocated == 30
        assert pool.total == 100

    def test_discrete_truncation(self):
        rt = ResourceType(name="gold", conservation=False, discrete=True)
        pool = ResourcePool(resource_type=rt)

        pool.set("p1", 10.7)
        assert pool.get("p1") == 10  # Truncated to int

    def test_continuous_resource(self):
        rt = ResourceType(name="mana", conservation=False, discrete=False)
        pool = ResourcePool(resource_type=rt)

        pool.set("p1", 10.7)
        assert pool.get("p1") == 10.7  # Kept as float

    def test_bounds(self):
        rt = ResourceType(name="health", conservation=False, discrete=False, min_value=0.0, max_value=100.0)
        pool = ResourcePool(resource_type=rt)

        pool.set("p1", 150.0)
        assert pool.get("p1") == 100.0  # Clamped to max

        pool.set("p1", -10.0)
        assert pool.get("p1") == 0.0  # Clamped to min

    def test_to_dict(self):
        rt = ResourceType(name="gold", conservation=True, discrete=True)
        pool = ResourcePool(resource_type=rt, holdings={"p1": 50, "p2": 30}, unallocated=20)
        d = pool.to_dict()
        assert d["resource"] == "gold"
        assert d["total"] == 100
