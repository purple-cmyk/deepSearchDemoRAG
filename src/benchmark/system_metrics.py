import threading
import time
from typing import Optional
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

def psutil_available() -> bool:
    return _PSUTIL_AVAILABLE

class SystemMetricsSampler:

    def __init__(self, interval: float=0.1, pid: Optional[int]=None):
        self._interval = interval
        self._pid = pid
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_samples: list = []
        self._rss_samples: list = []
        self.mean_cpu_percent: float = 0.0
        self.peak_cpu_percent: float = 0.0
        self.peak_rss_mb: float = 0.0
        self.sample_count: int = 0

    def _collect(self) -> None:
        if not _PSUTIL_AVAILABLE:
            return
        proc = psutil.Process(self._pid)
        while not self._stop.is_set():
            try:
                cpu = proc.cpu_percent(interval=None)
                rss = proc.memory_info().rss / (1024 * 1024)
                self._cpu_samples.append(cpu)
                self._rss_samples.append(rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self._interval)

    def __enter__(self) -> 'SystemMetricsSampler':
        if _PSUTIL_AVAILABLE:
            proc = psutil.Process(self._pid)
            proc.cpu_percent(interval=None)
            self._stop.clear()
            self._thread = threading.Thread(target=self._collect, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        if self._cpu_samples:
            self.mean_cpu_percent = sum(self._cpu_samples) / len(self._cpu_samples)
            self.peak_cpu_percent = max(self._cpu_samples)
            self.sample_count = len(self._cpu_samples)
        if self._rss_samples:
            self.peak_rss_mb = max(self._rss_samples)

def current_rss_mb() -> float:
    if not _PSUTIL_AVAILABLE:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 * 1024)
