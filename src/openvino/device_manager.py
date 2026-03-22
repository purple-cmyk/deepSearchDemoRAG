import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / 'configs' / 'settings.yaml'
PROJECT_ROOT = SETTINGS_PATH.parent.parent
ADAPTIVE = 'ADAPTIVE'


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        logger.warning('Settings file not found: %s', SETTINGS_PATH)
        return {}
    try:
        with open(SETTINGS_PATH, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning('Failed to read settings: %s', exc)
        return {}

class DeviceManager:

    def __init__(self, settings_override: Optional[Dict[str, Any]]=None):
        self._core = None
        self._devices: Optional[List[str]] = None
        self._settings: dict = settings_override if settings_override is not None else load_settings()
        self._initialise()

    def _initialise(self) -> None:
        try:
            try:
                from openvino import Core
            except ImportError:
                from openvino.runtime import Core
            self._core = Core()
            self._devices = self._core.available_devices
            logger.info('OpenVINO devices: %s', self._devices)
        except ImportError:
            logger.warning('openvino is not installed.  Device detection unavailable.  Install: pip install openvino')
            self._devices = []
        except Exception as exc:
            logger.error('Failed to initialise OpenVINO Core: %s', exc)
            self._devices = []

    @property
    def core(self):
        return self._core

    def list_devices(self) -> List[str]:
        return list(self._devices) if self._devices else []

    def select(self, preferred: str='CPU') -> str:
        devices = self.list_devices()
        if preferred.upper() == 'AUTO':
            if self._core is not None:
                logger.info('Selected device: AUTO (available devices: %s)', devices)
                return 'AUTO'
            logger.warning('AUTO requested but OpenVINO not installed, falling back to CPU')
            return 'CPU'
        if preferred.upper().startswith('MULTI:'):
            sub_devices = preferred.split(':')[1].split(',')
            valid_subs = [d for d in sub_devices if d in devices]
            if len(valid_subs) >= 2:
                multi_str = 'MULTI:' + ','.join(valid_subs)
                logger.info('Selected device: %s', multi_str)
                return multi_str
            elif valid_subs:
                logger.warning("MULTI requested but only '%s' available, using single device", valid_subs[0])
                return valid_subs[0]
            else:
                logger.warning('No MULTI sub-devices available, falling back to CPU')
                return 'CPU'
        if preferred in devices:
            logger.info('Selected device: %s', preferred)
            return preferred
        if 'CPU' in devices:
            logger.warning("Preferred device '%s' not available (have: %s). Falling back to CPU.", preferred, devices)
            return 'CPU'
        logger.error("No OpenVINO devices available.  Returning '%s' anyway (inference will fail unless openvino is installed).", preferred)
        return preferred

    def resolve_embedding_ir_path(self) -> Path:
        raw = self.get_embedding_model_path()
        if not raw:
            return Path()
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def _heuristic_preferred(self) -> str:
        prefer_order: List[str] = list(self._settings.get('openvino', {}).get('adaptive', {}).get('prefer_order', ['GPU', 'NPU', 'CPU']))
        devices = self.list_devices()
        for d in prefer_order:
            if d in devices:
                logger.info('Heuristic device: %s', d)
                return d
        return 'CPU' if 'CPU' in devices else (devices[0] if devices else 'CPU')

    def select_adaptive(self, model_xml: Path, n_iterations: int=3) -> str:
        adaptive = self._settings.get('openvino', {}).get('adaptive', {}) or {}
        if not adaptive.get('enabled', True):
            return self._heuristic_preferred()
        n_iter = int(adaptive.get('benchmark_iterations', n_iterations))
        prefer_order: List[str] = list(adaptive.get('prefer_order', ['GPU', 'NPU', 'CPU']))
        devices = self.list_devices()
        if not devices:
            return 'CPU'
        candidates = [d for d in prefer_order if d in devices]
        if not candidates:
            candidates = [d for d in devices if d != 'CPU'] + (['CPU'] if 'CPU' in devices else [])
        if not model_xml.exists() or self._core is None:
            for d in candidates:
                return d
            return 'CPU'
        try:
            import numpy as np
            model = self._core.read_model(str(model_xml))
            best_dev = None
            best_ms = float('inf')
            for device in candidates:
                try:
                    compiled = self._core.compile_model(model, device)
                    input_names = [inp.get_any_name() for inp in compiled.inputs]
                    seq_len = 32
                    batch_size = 1
                    inputs = {'input_ids': np.ones((batch_size, seq_len), dtype=np.int64), 'attention_mask': np.ones((batch_size, seq_len), dtype=np.int64)}
                    if 'token_type_ids' in input_names:
                        inputs['token_type_ids'] = np.zeros((batch_size, seq_len), dtype=np.int64)
                    for _ in range(2):
                        compiled(inputs)
                    times: List[float] = []
                    for _ in range(n_iter):
                        t0 = time.perf_counter()
                        compiled(inputs)
                        times.append((time.perf_counter() - t0) * 1000)
                    mean_ms = float(sum(times) / len(times))
                    logger.info('Adaptive benchmark %s: mean=%.2f ms', device, mean_ms)
                    if mean_ms < best_ms:
                        best_ms = mean_ms
                        best_dev = device
                except Exception as exc:
                    logger.warning('Adaptive benchmark skip %s: %s', device, exc)
            if best_dev:
                logger.info('Adaptive device selected: %s (%.2f ms)', best_dev, best_ms)
                return best_dev
        except Exception as exc:
            logger.warning('Adaptive benchmark failed: %s', exc)
        for d in candidates:
            return d
        return 'CPU'

    def select_embedding_device(self, model_xml_resolved: str) -> str:
        ov_settings = self._settings.get('openvino', {})
        preferred = ov_settings.get('device', 'CPU')
        adaptive_cfg = ov_settings.get('adaptive', {}) or {}
        path = Path(model_xml_resolved)
        if preferred.upper() == ADAPTIVE:
            return self.select_adaptive(path, n_iterations=int(adaptive_cfg.get('benchmark_iterations', 3)))
        return self.select_from_settings()

    def select_from_settings(self) -> str:
        ov_settings = self._settings.get('openvino', {})
        preferred = ov_settings.get('device', 'CPU')
        if preferred.upper() == ADAPTIVE:
            xml = self.resolve_embedding_ir_path()
            if xml.exists():
                return self.select_adaptive(xml)
            logger.warning('ADAPTIVE device requested but embedding IR missing — using CPU')
            return 'CPU'
        logger.info("Settings file requests device: '%s'", preferred)
        selected = self.select(preferred)
        if selected != preferred:
            logger.info("Device fallback: '%s' → '%s'", preferred, selected)
        return selected

    def is_openvino_enabled(self) -> bool:
        ov_settings = self._settings.get('openvino', {})
        return bool(ov_settings.get('enabled', False))

    def get_embedding_model_path(self) -> str:
        ov_settings = self._settings.get('openvino', {})
        return ov_settings.get('embedding_model_ir', '')

    def device_properties(self, device: str) -> Dict[str, str]:
        if self._core is None:
            return {'error': 'OpenVINO not installed'}
        props: Dict[str, str] = {}
        for key in ('FULL_DEVICE_NAME', 'DEVICE_ARCHITECTURE', 'OPTIMAL_NUMBER_OF_INFER_REQUESTS'):
            try:
                value = self._core.get_property(device, key)
                props[key] = str(value)
            except Exception:
                pass
        try:
            supported = self._core.get_property(device, 'SUPPORTED_PROPERTIES')
            props['SUPPORTED_PROPERTIES_COUNT'] = str(len(supported))
        except Exception:
            pass
        return props

    def device_summary(self) -> List[Dict[str, str]]:
        summaries = []
        for device in self.list_devices():
            props = self.device_properties(device)
            summaries.append({'device': device, 'name': props.get('FULL_DEVICE_NAME', 'Unknown'), 'architecture': props.get('DEVICE_ARCHITECTURE', ''), 'optimal_requests': props.get('OPTIMAL_NUMBER_OF_INFER_REQUESTS', '')})
        return summaries

    def benchmark_devices(self, model_xml: str, n_iterations: int=20, batch_size: int=1, seq_len: int=128) -> Dict[str, Dict[str, float]]:
        if self._core is None:
            return {'error': {'msg': 'OpenVINO not installed'}}
        import numpy as np
        xml_path = Path(model_xml)
        if not xml_path.exists():
            return {'error': {'msg': f'Model not found: {model_xml}'}}
        model = self._core.read_model(str(xml_path))
        results = {}
        for device in self.list_devices():
            try:
                logger.info('Benchmarking on %s...', device)
                compiled = self._core.compile_model(model, device)
                inputs = {'input_ids': np.ones((batch_size, seq_len), dtype=np.int64), 'attention_mask': np.ones((batch_size, seq_len), dtype=np.int64)}
                input_names = [inp.get_any_name() for inp in compiled.inputs]
                if 'token_type_ids' in input_names:
                    inputs['token_type_ids'] = np.zeros((batch_size, seq_len), dtype=np.int64)
                for _ in range(3):
                    compiled(inputs)
                times = []
                for _ in range(n_iterations):
                    start = time.perf_counter()
                    compiled(inputs)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed * 1000)
                times_arr = np.array(times)
                results[device] = {'mean_ms': float(times_arr.mean()), 'std_ms': float(times_arr.std()), 'min_ms': float(times_arr.min()), 'max_ms': float(times_arr.max()), 'median_ms': float(np.median(times_arr))}
                logger.info('%s: mean=%.2fms, min=%.2fms, max=%.2fms', device, results[device]['mean_ms'], results[device]['min_ms'], results[device]['max_ms'])
            except Exception as exc:
                logger.warning('Benchmark failed on %s: %s', device, exc)
                results[device] = {'error': str(exc)}
        return results
