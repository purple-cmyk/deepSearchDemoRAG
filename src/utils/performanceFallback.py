import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.defaults import EMBEDDING_MODEL_ID
logger = logging.getLogger(__name__)


def _probe_encoder_ok(encoder: Any) -> bool:
    try:
        v = encoder.encode(['__probe__'], batch_size=1, show_progress=False)
        if v is None or v.size == 0:
            return False
        return float(np.abs(v).sum()) > 1e-8
    except Exception as exc:
        logger.warning('Encoder probe failed: %s', exc)
        return False


def create_embedding_encoder(project_root: Path, settings: Dict[str, Any], device_override: Optional[str]=None) -> Tuple[Any, Dict[str, Any]]:
    from src.embeddings.encoder import EmbeddingEncoder
    from src.embeddings.openvino_encoder import OVEmbeddingEncoder
    from src.openvino.device_manager import DeviceManager
    emb_cfg = settings.get('embeddings', {}) or {}
    emb_name = emb_cfg.get('model_name') or EMBEDDING_MODEL_ID
    emb_device = emb_cfg.get('device', 'cpu')
    meta: Dict[str, Any] = {'backend': 'pytorch', 'device': emb_device, 'fallback_reason': None}
    dm = DeviceManager(settings_override=settings)
    if not dm.is_openvino_enabled():
        enc = EmbeddingEncoder(model_name=emb_name, device=emb_device)
        return enc, meta
    raw_ir = dm.get_embedding_model_path()
    if not raw_ir:
        enc = EmbeddingEncoder(model_name=emb_name, device=emb_device)
        return enc, meta
    model_ir = Path(raw_ir)
    if not model_ir.is_absolute():
        model_ir = project_root / model_ir
    if not model_ir.exists():
        logger.warning('OpenVINO IR not found at %s — using PyTorch encoder', model_ir)
        meta['fallback_reason'] = 'ir_missing'
        enc = EmbeddingEncoder(model_name=emb_name, device=emb_device)
        return enc, meta
    if device_override:
        ov_device = dm.select(device_override)
    else:
        ov_device = dm.select_embedding_device(str(model_ir))
    try:
        ov_enc = OVEmbeddingEncoder(model_xml=str(model_ir), device=ov_device, tokenizer_name=emb_name)
        if _probe_encoder_ok(ov_enc):
            meta.update({'backend': 'openvino', 'device': ov_device, 'model_xml': str(model_ir)})
            return ov_enc, meta
        logger.warning('OpenVINO encoder produced invalid output — falling back to PyTorch')
        meta['fallback_reason'] = 'probe_failed'
    except Exception as exc:
        logger.warning('OpenVINO encoder init failed (%s) — falling back to PyTorch', exc)
        meta['fallback_reason'] = str(exc)[:200]
    enc = EmbeddingEncoder(model_name=emb_name, device=emb_device)
    return enc, meta
