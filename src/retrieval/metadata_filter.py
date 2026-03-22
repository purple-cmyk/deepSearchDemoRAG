import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
logger = logging.getLogger(__name__)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class MetadataStore:
    _EXT_TO_TYPE: Dict[str, str] = {'.pdf': 'pdf', '.docx': 'docx', '.doc': 'docx', '.pptx': 'pptx', '.ppt': 'pptx', '.txt': 'text', '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.tiff': 'image', '.tif': 'image', '.bmp': 'image', '.mp4': 'video', '.avi': 'video', '.mkv': 'video', '.mov': 'video'}
    _TYPE_KEYWORDS: Dict[str, str] = {'pdf': 'pdf', 'docx': 'docx', 'pptx': 'pptx', 'slides': 'pptx', 'presentation': 'pptx', 'video': 'video', 'image': 'image', 'report': 'pdf', 'document': 'pdf', 'form': 'image', 'text': 'text'}

    def __init__(self, chunk_metadata: List[Dict]):
        self._entries: List[Dict] = []
        self._size = len(chunk_metadata)
        for idx, raw in enumerate(chunk_metadata):
            enriched = self._enrich(idx, raw)
            self._entries.append(enriched)

    @property
    def size(self) -> int:
        return self._size

    def get(self, idx: int) -> Dict:
        if 0 <= idx < self._size:
            return self._entries[idx]
        return {}

    def _enrich(self, idx: int, raw: Dict) -> Dict:
        meta = raw.get('metadata', {})
        source = raw.get('source', raw.get('doc_id', ''))
        file_name = meta.get('file_name', '')
        if not file_name:
            for candidate in [raw.get('image_path', ''), source]:
                if candidate and os.sep in str(candidate):
                    file_name = Path(str(candidate)).name
                    break
            if not file_name:
                file_name = raw.get('doc_id', f'chunk_{idx}')
        file_type = meta.get('file_type', '')
        if not file_type:
            ext = Path(file_name).suffix.lower() if file_name else ''
            file_type = self._EXT_TO_TYPE.get(ext, '')
            if not file_type:
                doc_type = raw.get('doc_type', meta.get('doc_type', ''))
                dataset = meta.get('dataset', '')
                if doc_type in ('form', 'document_image', 'classified_image', 'image'):
                    file_type = 'image'
                elif dataset in ('funsd', 'docvqa', 'rvl_cdip'):
                    file_type = 'image'
                else:
                    file_type = 'unknown'
        source_directory = meta.get('source_directory', '')
        if not source_directory:
            for candidate in [raw.get('image_path', ''), source]:
                if candidate and os.sep in str(candidate):
                    source_directory = str(Path(str(candidate)).parent)
                    break
        created_year = meta.get('created_year')
        page_number = meta.get('page_number')
        slide_number = meta.get('slide_number')
        modality = meta.get('modality', '')
        if not modality:
            if file_type in ('mp4', 'avi', 'mkv', 'mov', 'video'):
                modality = 'video'
            elif file_type in ('image', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'):
                modality = 'image'
            else:
                modality = 'text'
        return {'file_name': file_name, 'file_type': file_type, 'created_year': created_year, 'source_directory': source_directory, 'page_number': page_number, 'slide_number': slide_number, 'modality': modality, 'doc_id': raw.get('doc_id', ''), 'chunk_id': raw.get('chunk_id', ''), 'dataset': meta.get('dataset', '')}

    def filter_ids(self, constraints: Dict[str, Any]) -> Optional[List[int]]:
        if not constraints:
            return None
        matching: List[int] = []
        for idx, entry in enumerate(self._entries):
            if self._matches(entry, constraints):
                matching.append(idx)
        logger.info('Metadata filter: %d/%d chunks matched constraints %s', len(matching), self._size, constraints)
        return matching

    @staticmethod
    def _matches(entry: Dict, constraints: Dict[str, Any]) -> bool:
        for key, value in constraints.items():
            if value is None:
                continue
            if key == 'year':
                entry_year = entry.get('created_year')
                if entry_year is None:
                    continue
                if str(entry_year) != str(value):
                    return False
            elif key == 'file_type':
                entry_val = (entry.get('file_type') or '').lower()
                if str(value).lower() not in entry_val:
                    return False
            elif key == 'directory':
                entry_val = (entry.get('source_directory') or '').lower()
                if str(value).lower() not in entry_val:
                    return False
            elif key == 'modality':
                entry_val = (entry.get('modality') or '').lower()
                if str(value).lower() != entry_val:
                    return False
            elif key == 'dataset':
                entry_val = (entry.get('dataset') or '').lower()
                if str(value).lower() not in entry_val:
                    return False
        return True

class QueryMetadataParser:
    _YEAR_RE = re.compile('\\b((?:19|20)\\d{2})\\b')
    _TYPE_MAP: Dict[str, str] = {'pdf': 'pdf', 'pdfs': 'pdf', 'docx': 'docx', 'word': 'docx', 'pptx': 'pptx', 'ppt': 'pptx', 'slides': 'pptx', 'slide': 'pptx', 'presentation': 'pptx', 'presentations': 'pptx', 'video': 'video', 'videos': 'video', 'clip': 'video', 'clips': 'video', 'image': 'image', 'images': 'image', 'photo': 'image', 'photos': 'image', 'picture': 'image', 'pictures': 'image', 'report': 'pdf', 'reports': 'pdf', 'document': 'pdf', 'documents': 'pdf', 'form': 'image', 'forms': 'image', 'invoice': 'image', 'invoices': 'image', 'text': 'text'}
    _MODALITY_MAP: Dict[str, str] = {'video': 'video', 'videos': 'video', 'clip': 'video', 'clips': 'video', 'image': 'image', 'images': 'image', 'photo': 'image', 'picture': 'image', 'text': 'text', 'document': 'text'}
    _DIR_RE = re.compile('(?:folder|dir|directory|from|in)\\s*:\\s*[\\"\']?([^\\s\\"\',]+)[\\"\']?', re.IGNORECASE)

    def parse(self, query: str) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {'year': None, 'file_type': None, 'directory': None, 'modality': None}
        if not query:
            return constraints
        lower = query.lower()
        tokens = re.findall('\\w+', lower)
        year_match = self._YEAR_RE.search(query)
        if year_match:
            constraints['year'] = int(year_match.group(1))
        for token in tokens:
            if token in self._TYPE_MAP:
                constraints['file_type'] = self._TYPE_MAP[token]
                break
        dir_match = self._DIR_RE.search(query)
        if dir_match:
            constraints['directory'] = dir_match.group(1)
        for token in tokens:
            if token in self._MODALITY_MAP:
                constraints['modality'] = self._MODALITY_MAP[token]
                break
        constraints = {k: v for k, v in constraints.items() if v is not None}
        if constraints:
            logger.info('Parsed metadata hints from query: %s', constraints)
        return constraints

class StagedRetriever:

    def __init__(self, encoder, index, metadata_store: Optional[MetadataStore]=None, parser: Optional[QueryMetadataParser]=None):
        self.encoder = encoder
        self.index = index
        self.metadata_store = metadata_store
        self.parser = parser or QueryMetadataParser()
        self.last_search_stats: Dict[str, Any] = {}

    def query(self, query_text: str, top_k: int=5, constraints: Optional[Dict[str, Any]]=None, auto_parse: bool=True, score_threshold: Optional[float]=None) -> Tuple[List[Dict], Dict[str, Any]]:
        stats: Dict[str, Any] = {'mode': 'metadata-aware', 'query': query_text, 'constraints_detected': {}, 'candidates_before_filter': 0, 'candidates_after_filter': 0, 'used_prefilter': False, 'retrieval_latency_ms': 0.0, 'encoding_latency_ms': 0.0, 'filter_latency_ms': 0.0, 'search_latency_ms': 0.0, 'fallback_to_full': False}
        t_total_start = time.perf_counter()
        if constraints is None and auto_parse:
            constraints = self.parser.parse(query_text)
        stats['constraints_detected'] = constraints or {}
        total_vectors = self.index.size if self.index.index is not None else 0
        stats['candidates_before_filter'] = total_vectors
        t_enc_start = time.perf_counter()
        query_vector = self.encoder.encode_single(query_text, normalize=True)
        stats['encoding_latency_ms'] = (time.perf_counter() - t_enc_start) * 1000
        filtered_ids: Optional[List[int]] = None
        if constraints and self.metadata_store is not None:
            t_filter_start = time.perf_counter()
            filtered_ids = self.metadata_store.filter_ids(constraints)
            stats['filter_latency_ms'] = (time.perf_counter() - t_filter_start) * 1000
            if filtered_ids is not None:
                stats['candidates_after_filter'] = len(filtered_ids)
                if len(filtered_ids) == 0:
                    logger.info('Pre-filter returned 0 candidates; falling back to full search')
                    filtered_ids = None
                    stats['fallback_to_full'] = True
                else:
                    stats['used_prefilter'] = True
        else:
            stats['candidates_after_filter'] = total_vectors
        t_search_start = time.perf_counter()
        if filtered_ids is not None and FAISS_AVAILABLE:
            results = self._search_with_ids(query_vector, filtered_ids, top_k)
        else:
            results = self.index.search(query_vector, top_k=top_k)
        stats['search_latency_ms'] = (time.perf_counter() - t_search_start) * 1000
        if score_threshold is not None:
            results = [r for r in results if r.get('score', 0) >= score_threshold]
        stats['retrieval_latency_ms'] = (time.perf_counter() - t_total_start) * 1000
        stats['results_count'] = len(results)
        self.last_search_stats = stats
        return (results, stats)

    def _search_with_ids(self, query_vector: np.ndarray, candidate_ids: List[int], top_k: int) -> List[Dict]:
        if self.index.index is None or not candidate_ids:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        effective_top_k = min(top_k, len(candidate_ids))
        try:
            id_array = np.array(candidate_ids, dtype=np.int64)
            selector = faiss.IDSelectorBatch(id_array)
            params = faiss.SearchParametersIVF()
            params.sel = selector
            scores, indices = self.index.index.search(query_vector, effective_top_k, params=params)
        except (AttributeError, TypeError, RuntimeError):
            logger.debug('FAISS IDSelectorBatch not supported; using Python-side filtering')
            return self._search_fallback(query_vector, candidate_ids, top_k)
        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = {**self.index.metadata[idx], 'score': float(score)}
            results.append(entry)
        return results

    def _search_fallback(self, query_vector: np.ndarray, candidate_ids: List[int], top_k: int) -> List[Dict]:
        fetch_k = min(self.index.size, max(top_k * 10, len(candidate_ids)))
        all_results = self.index.search(query_vector, top_k=fetch_k)
        candidate_set = set(candidate_ids)
        filtered: List[Dict] = []
        for entry in all_results:
            chunk_id = entry.get('chunk_id', '')
            idx = self._find_index_for_chunk(chunk_id)
            if idx is not None and idx in candidate_set:
                filtered.append(entry)
                if len(filtered) >= top_k:
                    break
        return filtered

    def _find_index_for_chunk(self, chunk_id: str) -> Optional[int]:
        for i, meta in enumerate(self.index.metadata):
            if meta.get('chunk_id') == chunk_id:
                return i
        return None

def enrich_chunk_metadata(chunk_meta: Dict, file_name: str='', file_type: str='', created_year: Optional[int]=None, source_directory: str='', page_number: Optional[int]=None, slide_number: Optional[int]=None, modality: str='text') -> Dict:
    inner = chunk_meta.get('metadata', chunk_meta)
    if file_name and (not inner.get('file_name')):
        inner['file_name'] = file_name
    if file_type and (not inner.get('file_type')):
        inner['file_type'] = file_type
    if created_year is not None and (not inner.get('created_year')):
        inner['created_year'] = created_year
    if source_directory and (not inner.get('source_directory')):
        inner['source_directory'] = source_directory
    if page_number is not None and inner.get('page_number') is None:
        inner['page_number'] = page_number
    if slide_number is not None and inner.get('slide_number') is None:
        inner['slide_number'] = slide_number
    if modality and (not inner.get('modality')):
        inner['modality'] = modality
    return chunk_meta
