"""Observability — structured logging + lightweight metrics.

Production simulations need to answer:
  - How fast does the engine run? (rounds_per_second)
  - What's the latency distribution of action resolution?
  - Which effect ops are hot?
  - How often do triggers cascade?

This module provides a **zero-cost-when-disabled** primitive layer:

  - ``Counter`` — monotonically incrementing values
  - ``Histogram`` — distribution of observed values (latencies, sizes)
  - ``Gauge`` — point-in-time values that can go up or down
  - ``timed(name)`` — context manager that records elapsed ms into a Histogram

All metrics live in a single ``MetricsRegistry`` that any caller can
inspect, snapshot, or export. The default registry is process-global;
tests use scoped registries via ``with metrics_registry.scoped()``.

## Structured logging

``get_logger(name)`` returns a stdlib Logger configured for JSON-line
output when the env var ``KERNEL_LOG_JSON=1`` is set, otherwise
human-readable. Use it in place of plain ``logging.getLogger`` so log
events have a uniform structure across the kernel.

## Engine instrumentation

The engine's hot paths emit metrics through this module. They're
disabled by default; turn them on with ``enable_engine_metrics()`` or
the env var ``KERNEL_METRICS=1``. Disabled metrics are cheap (a single
boolean check).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


class Counter:
    """Monotonically incrementing integer. Thread-safe."""
    __slots__ = ("_value", "_lock", "name", "labels")

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = dict(labels or {})
        self._value: int = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    def value(self) -> int:
        return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0


class Gauge:
    """Point-in-time value that can go up or down. Thread-safe."""
    __slots__ = ("_value", "_lock", "name", "labels")

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = dict(labels or {})
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, v: float) -> None:
        with self._lock:
            self._value = float(v)

    def inc(self, n: float = 1.0) -> None:
        with self._lock:
            self._value += n

    def dec(self, n: float = 1.0) -> None:
        with self._lock:
            self._value -= n

    def value(self) -> float:
        return self._value


class Histogram:
    """Distribution of observed values. Tracks count, sum, min, max,
    and p50/p95/p99 percentiles over the last 10k samples (ring buffer)."""
    __slots__ = ("_samples", "_count", "_sum", "_min", "_max", "_lock",
                 "name", "labels", "_max_samples")

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None,
                 max_samples: int = 10000):
        self.name = name
        self.labels = dict(labels or {})
        self._samples: List[float] = []
        self._count: int = 0
        self._sum: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._max_samples = max_samples
        self._lock = threading.Lock()

    def observe(self, v: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += v
            if v < self._min:
                self._min = v
            if v > self._max:
                self._max = v
            self._samples.append(v)
            if len(self._samples) > self._max_samples:
                # Ring buffer — drop oldest half
                drop = len(self._samples) - self._max_samples
                self._samples = self._samples[drop:]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            if self._count == 0:
                return {"count": 0}
            ordered = sorted(self._samples)
            n = len(ordered)
            return {
                "count": self._count,
                "sum": self._sum,
                "min": self._min,
                "max": self._max,
                "avg": self._sum / self._count,
                "p50": ordered[n // 2],
                "p95": ordered[min(n - 1, int(n * 0.95))],
                "p99": ordered[min(n - 1, int(n * 0.99))],
            }


# ---------------------------------------------------------------------------
# Registry — collects every Counter/Gauge/Histogram by name
# ---------------------------------------------------------------------------


class MetricsRegistry:
    """Process-wide metric registry.

    All metrics are accessible by name. Snapshot via ``export()`` for
    monitoring integrations. Use ``scoped()`` in tests to isolate."""

    def __init__(self, parent: Optional["MetricsRegistry"] = None):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._parent = parent

    def counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> Counter:
        key = self._key(name, labels)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(name, labels)
            return self._counters[key]

    def gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Gauge:
        key = self._key(name, labels)
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge(name, labels)
            return self._gauges[key]

    def histogram(self, name: str, labels: Optional[Dict[str, str]] = None) -> Histogram:
        key = self._key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(name, labels)
            return self._histograms[key]

    def export(self) -> Dict[str, Any]:
        """Snapshot all metrics for monitoring export."""
        with self._lock:
            return {
                "counters": {k: c.value() for k, c in self._counters.items()},
                "gauges": {k: g.value() for k, g in self._gauges.items()},
                "histograms": {k: h.stats() for k, h in self._histograms.items()},
            }

    def reset(self) -> None:
        """Reset all metrics. Useful in tests."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    @contextlib.contextmanager
    def scoped(self) -> Iterator["MetricsRegistry"]:
        """Yield a child registry. Lets tests collect metrics without
        polluting the process-wide one. (Child doesn't currently
        fall back to parent — keep them fully isolated.)"""
        yield MetricsRegistry(parent=self)

    @staticmethod
    def _key(name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Process-wide singleton
metrics = MetricsRegistry()


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def timed(name: str, labels: Optional[Dict[str, str]] = None,
          registry: Optional[MetricsRegistry] = None) -> Iterator[None]:
    """Context manager: record elapsed ms into a Histogram.

        with timed("apply_effects_ms"):
            apply_effects(engine, ...)
    """
    reg = registry or metrics
    hist = reg.histogram(name, labels)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        hist.observe(elapsed_ms)


# ---------------------------------------------------------------------------
# Feature toggle — engine metrics enabled/disabled
# ---------------------------------------------------------------------------


_engine_metrics_enabled: bool = os.environ.get("KERNEL_METRICS", "0") == "1"


def enable_engine_metrics(enabled: bool = True) -> None:
    """Toggle engine-level metric recording at runtime.

    When disabled (the default), the engine's instrumentation points
    are cheap boolean checks. When enabled, hot paths record into
    histograms/counters. Override with env var ``KERNEL_METRICS=1`` or
    call this function before constructing your engine.
    """
    global _engine_metrics_enabled
    _engine_metrics_enabled = bool(enabled)


def engine_metrics_enabled() -> bool:
    return _engine_metrics_enabled


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Allow arbitrary structured fields via logger.info("...", extra={"key": val})
        for k, v in record.__dict__.items():
            if k in ("name", "msg", "args", "levelname", "levelno", "pathname",
                     "filename", "module", "exc_info", "exc_text", "stack_info",
                     "lineno", "funcName", "created", "msecs", "relativeCreated",
                     "thread", "threadName", "processName", "process", "message"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        return json.dumps(payload, default=str)


_log_handler_installed = False


def _ensure_handler() -> None:
    global _log_handler_installed
    if _log_handler_installed:
        return
    _log_handler_installed = True
    root_logger = logging.getLogger("fg_env_kernel")
    if os.environ.get("KERNEL_LOG_JSON", "0") == "1":
        h = logging.StreamHandler()
        h.setFormatter(_JsonFormatter())
        root_logger.addHandler(h)
    # Otherwise let the application's default logging config handle it.


def get_logger(name: str) -> logging.Logger:
    """Return a kernel-namespaced logger. JSON-formatted when
    ``KERNEL_LOG_JSON=1`` is set, otherwise plain text."""
    _ensure_handler()
    return logging.getLogger(name)


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "metrics",
    "timed",
    "enable_engine_metrics",
    "engine_metrics_enabled",
    "get_logger",
]
