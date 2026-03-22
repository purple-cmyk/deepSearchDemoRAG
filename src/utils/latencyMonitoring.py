import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

_lock = threading.Lock()
_instance: Optional['LatencyMonitor'] = None


@dataclass
class StageStats:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


class LatencyMonitor:

    def __init__(self) -> None:
        self._stages: Dict[str, StageStats] = {}
        self.enabled: bool = True

    def configure(self, enabled: bool) -> None:
        self.enabled = enabled

    def record(self, stage: str, duration_ms: float) -> None:
        if not self.enabled:
            return
        with _lock:
            if stage not in self._stages:
                self._stages[stage] = StageStats()
            self._stages[stage].record(duration_ms)

    def reset(self) -> None:
        with _lock:
            self._stages.clear()

    def summary(self) -> Dict[str, Dict[str, float]]:
        with _lock:
            out: Dict[str, Dict[str, float]] = {}
            for name, st in self._stages.items():
                out[name] = {'count': float(st.count), 'mean_ms': round(st.mean_ms, 3), 'max_ms': round(st.max_ms, 3), 'total_ms': round(st.total_ms, 3)}
            return out

    def wrap_encoder(self, inner: Any, prefix: str='embed') -> Any:
        return _MonitoredEncoder(inner, self, prefix)


def get_latency_monitor() -> LatencyMonitor:
    global _instance
    if _instance is None:
        _instance = LatencyMonitor()
    return _instance


@contextmanager
def timed_stage(stage: str) -> Iterator[None]:
    mon = get_latency_monitor()
    if not mon.enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        mon.record(stage, (time.perf_counter() - t0) * 1000.0)


class _MonitoredEncoder:

    def __init__(self, inner: Any, monitor: LatencyMonitor, prefix: str) -> None:
        object.__setattr__(self, '_inner', inner)
        object.__setattr__(self, '_monitor', monitor)
        object.__setattr__(self, '_prefix', prefix)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return self._inner.encode(*args, **kwargs)
        finally:
            self._monitor.record(f'{self._prefix}.encode', (time.perf_counter() - t0) * 1000.0)

    def encode_single(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return self._inner.encode_single(*args, **kwargs)
        finally:
            self._monitor.record(f'{self._prefix}.encode_single', (time.perf_counter() - t0) * 1000.0)
