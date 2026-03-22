import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
logger = logging.getLogger(__name__)

class MultimodalRetriever:

    def __init__(self, text_encoder, text_index, clip_encoder=None, clip_index=None, text_weight: float=0.6, clip_weight: float=0.4):
        self.text_encoder = text_encoder
        self.text_index = text_index
        self.clip_encoder = clip_encoder
        self.clip_index = clip_index
        self.text_weight = text_weight
        self.clip_weight = clip_weight

    @property
    def has_clip(self) -> bool:
        return self.clip_encoder is not None and self.clip_index is not None and getattr(self.clip_encoder, 'is_available', False)

    def query(self, query_text: str, top_k: int=5) -> List[Dict]:
        text_results = self._text_search(query_text, top_k=top_k * 2)
        clip_results = []
        if self.has_clip:
            clip_results = self._clip_search(query_text, top_k=top_k * 2)
        if clip_results:
            merged = self._fuse_results(text_results, clip_results, top_k)
        else:
            merged = text_results[:top_k]
        logger.info("Multimodal query '%s': %d text + %d clip → %d fused", query_text[:60], len(text_results), len(clip_results), len(merged))
        return merged

    def _text_search(self, query_text: str, top_k: int) -> List[Dict]:
        try:
            query_vec = self.text_encoder.encode_single(query_text, normalize=True)
            results = self.text_index.search(query_vec, top_k=top_k)
            for r in results:
                r['retrieval_source'] = 'text'
            return results
        except Exception as exc:
            logger.error('Text retrieval failed: %s', exc)
            return []

    def _clip_search(self, query_text: str, top_k: int) -> List[Dict]:
        try:
            clip_vec = self.clip_encoder.encode_text(query_text)
            if clip_vec is None:
                return []
            results = self.clip_index.search(clip_vec, top_k=top_k)
            for r in results:
                r['retrieval_source'] = 'clip'
            return results
        except Exception as exc:
            logger.error('CLIP retrieval failed: %s', exc)
            return []

    def _fuse_results(self, text_results: List[Dict], clip_results: List[Dict], top_k: int) -> List[Dict]:
        text_results = self._normalise_scores(text_results)
        clip_results = self._normalise_scores(clip_results)
        fused: Dict[str, Dict] = {}
        for r in text_results:
            key = r.get('chunk_id', r.get('doc_id', id(r)))
            if key not in fused:
                fused[key] = {**r, 'fused_score': 0.0}
            fused[key]['fused_score'] += r['score'] * self.text_weight
        for r in clip_results:
            key = r.get('chunk_id', r.get('doc_id', id(r)))
            if key not in fused:
                fused[key] = {**r, 'fused_score': 0.0}
            fused[key]['fused_score'] += r['score'] * self.clip_weight
        sorted_results = sorted(fused.values(), key=lambda x: x['fused_score'], reverse=True)
        for r in sorted_results:
            r['score'] = r.pop('fused_score')
        return sorted_results[:top_k]

    @staticmethod
    def _normalise_scores(results: List[Dict]) -> List[Dict]:
        if not results:
            return results
        scores = [r['score'] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s
        if span < 1e-09:
            for r in results:
                r['score'] = 1.0
        else:
            for r in results:
                r['score'] = (r['score'] - min_s) / span
        return results
