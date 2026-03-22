import copy
import os
from typing import Any, Dict


def is_edge_mode(settings: Dict[str, Any]) -> bool:
    if os.environ.get('DEEP_SEARCH_EDGE', '').strip() in ('1', 'true', 'yes', 'on'):
        return True
    rt = settings.get('runtime', {}) if isinstance(settings, dict) else {}
    return bool(rt.get('edge_mode', False))


def apply_edge_overrides(settings: Dict[str, Any]) -> Dict[str, Any]:
    if not is_edge_mode(settings):
        return settings
    out = copy.deepcopy(settings)
    emb = out.setdefault('embeddings', {})
    emb['batch_size'] = min(int(emb.get('batch_size', 64)), 8)
    ret = out.setdefault('retrieval', {})
    ret['top_k'] = min(int(ret.get('top_k', 5)), 4)
    clip = out.setdefault('clip', {})
    clip['enabled'] = False
    vid = out.setdefault('video', {})
    vid['enable_frame_captioning'] = False
    vid['enable_whisper'] = False
    vid['caption_interval'] = max(int(vid.get('caption_interval', 5)), 10)
    llm = out.setdefault('llm', {})
    llm['max_tokens'] = min(int(llm.get('max_tokens', 512)), 256)
    ov = out.setdefault('openvino', {})
    if ov.get('device', 'CPU') not in ('CPU',):
        ov['device'] = 'CPU'
    return out
