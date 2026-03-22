import logging
import re
import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, List, Dict, Optional, Set
import numpy as np
from src.defaults import CROSS_ENCODER_MODEL_ID
from src.embeddings.encoder import EmbeddingEncoder
from src.index.faiss_index import FaissIndex
from src.runtime.latency_monitor import timed_stage
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from src.retrieval.query_router import QueryPlan
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

class RetrieverResult:

    def __init__(self, chunk_id: str, doc_id: str, text: str, score: float, metadata: Dict):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        preview = self.text[:80].replace('\n', ' ')
        return f"RetrieverResult(score={self.score:.4f}, chunk='{preview}...')"

    def to_dict(self) -> Dict:
        return {'chunk_id': self.chunk_id, 'doc_id': self.doc_id, 'text': self.text, 'score': self.score, 'metadata': self.metadata}

class QueryPreprocessor:
    SPELLING_FIXES: Dict[str, str] = {'invoce': 'invoice', 'invioce': 'invoice', 'reciept': 'receipt', 'recieve': 'receive', 'adress': 'address', 'ammount': 'amount', 'amoutn': 'amount', 'signiture': 'signature', 'employe': 'employee', 'documnet': 'document', 'infomation': 'information', 'sumary': 'summary', 'purchse': 'purchase', 'compnay': 'company', 'accout': 'account', 'totla': 'total', 'fomr': 'form'}
    SYNONYMS: Dict[str, List[str]] = {'invoice': ['bill', 'receipt', 'statement'], 'employee': ['worker', 'staff', 'personnel'], 'amount': ['total', 'sum', 'value', 'price'], 'address': ['location', 'place'], 'company': ['organisation', 'firm', 'business'], 'date': ['day', 'time', 'when'], 'sign': ['signature', 'signed', 'autograph'], 'name': ['person', 'who', 'identity'], 'purchase': ['buy', 'order', 'procurement']}

    def preprocess(self, query: str, fix_spelling: bool=True, expand_synonyms: bool=True) -> str:
        text = query.strip()
        text = re.sub('\\s+', ' ', text)
        text = text.lower()
        if fix_spelling:
            words = text.split()
            words = [self.SPELLING_FIXES.get(w, w) for w in words]
            text = ' '.join(words)
        if expand_synonyms:
            expansions: List[str] = []
            for key, syns in self.SYNONYMS.items():
                if key in text:
                    expansions.extend(syns)
            if expansions:
                text = text + ' ' + ' '.join(expansions)
        logger.debug("Query preprocessed: '%s' -> '%s'", query, text)
        return text

class BM25Retriever:

    def __init__(self, k1: float=1.5, b: float=0.75):
        self.k1 = k1
        self.b = b
        self._corpus_tokens: List[List[str]] = []
        self._doc_lengths: List[int] = []
        self._avgdl: float = 0.0
        self._n_docs: int = 0
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._metadata: List[Dict] = []

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall('\\w+', text.lower())

    def fit(self, documents: List[Dict]) -> None:
        self._metadata = documents
        self._corpus_tokens = []
        self._df = defaultdict(int)
        for doc in documents:
            tokens = self._tokenise(doc.get('text', ''))
            self._corpus_tokens.append(tokens)
            for term in set(tokens):
                self._df[term] += 1
        self._n_docs = len(documents)
        self._doc_lengths = [len(t) for t in self._corpus_tokens]
        self._avgdl = sum(self._doc_lengths) / self._n_docs if self._n_docs else 1.0
        for term, df_val in self._df.items():
            self._idf[term] = math.log((self._n_docs - df_val + 0.5) / (df_val + 0.5) + 1.0)
        logger.info('BM25 fitted: %d documents, %d unique terms, avgdl=%.1f', self._n_docs, len(self._df), self._avgdl)

    def search(self, query: str, top_k: int=10) -> List[Dict]:
        query_tokens = self._tokenise(query)
        if not query_tokens or not self._corpus_tokens:
            return []
        scores = np.zeros(self._n_docs, dtype=np.float64)
        for qt in query_tokens:
            idf = self._idf.get(qt, 0.0)
            if idf == 0.0:
                continue
            for i, doc_tokens in enumerate(self._corpus_tokens):
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                dl = self._doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * (numerator / denominator)
        top_k = min(top_k, self._n_docs)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            entry = {**self._metadata[idx], 'score': float(scores[idx])}
            results.append(entry)
        return results

class CrossEncoderReranker:
    DEFAULT_MODEL = CROSS_ENCODER_MODEL_ID

    def __init__(self, model_name: str=DEFAULT_MODEL, device: str='cpu'):
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError('sentence-transformers is required for cross-encoder re-ranking.  Install: pip install sentence-transformers')
        logger.info('Loading cross-encoder: %s on %s', model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(self, query: str, results: List['RetrieverResult'], top_k: Optional[int]=None) -> List['RetrieverResult']:
        if not results:
            return []
        pairs = [(query, r.text) for r in results]
        ce_scores = self.model.predict(pairs)
        scored: List[tuple] = sorted(zip(ce_scores, results), key=lambda x: x[0], reverse=True)
        reranked: List[RetrieverResult] = []
        for score, result in scored:
            new_result = RetrieverResult(chunk_id=result.chunk_id, doc_id=result.doc_id, text=result.text, score=float(score), metadata={**result.metadata, 'original_score': result.score})
            reranked.append(new_result)
        if top_k is not None:
            reranked = reranked[:top_k]
        logger.info('Cross-encoder re-ranked %d -> %d results', len(results), len(reranked))
        return reranked

class Retriever:

    def __init__(self, encoder: EmbeddingEncoder, index: FaissIndex, enable_bm25: bool=False, enable_reranker: bool=False, enable_preprocessing: bool=False, reranker_model: str=CrossEncoderReranker.DEFAULT_MODEL, reranker_device: str='cpu'):
        self.encoder = encoder
        self.index = index
        self._preprocessor: Optional[QueryPreprocessor] = None
        if enable_preprocessing:
            self._preprocessor = QueryPreprocessor()
            logger.info('Query preprocessing enabled')
        self._bm25: Optional[BM25Retriever] = None
        if enable_bm25 and index.metadata:
            self._bm25 = BM25Retriever()
            self._bm25.fit(index.metadata)
            logger.info('BM25 hybrid retrieval enabled (%d docs)', len(index.metadata))
        self._reranker: Optional[CrossEncoderReranker] = None
        if enable_reranker:
            self._reranker = CrossEncoderReranker(model_name=reranker_model, device=reranker_device)
            logger.info('Cross-encoder re-ranking enabled')

    def query(self, query_text: str, top_k: int=5, score_threshold: Optional[float]=None, filters: Optional[Dict[str, str]]=None, use_hybrid: Optional[bool]=None, use_reranker: Optional[bool]=None, use_preprocessing: Optional[bool]=None, routing_plan: Optional['QueryPlan']=None) -> List[RetrieverResult]:
        if not query_text.strip():
            logger.warning('Empty query, returning no results')
            return []
        effective_top_k = routing_plan.top_k if routing_plan else top_k
        do_preprocess = use_preprocessing if use_preprocessing is not None else self._preprocessor is not None
        if routing_plan and routing_plan.use_preprocessing:
            do_preprocess = True
        if do_preprocess:
            preprocessor = self._preprocessor or QueryPreprocessor()
            processed_query = preprocessor.preprocess(query_text)
            logger.info("Preprocessed query: '%s' -> '%s'", query_text[:60], processed_query[:80])
        else:
            processed_query = query_text
        do_hybrid = use_hybrid if use_hybrid is not None else self._bm25 is not None
        if routing_plan:
            do_hybrid = routing_plan.use_hybrid and self._bm25 is not None
        dense_mult = 3
        if routing_plan:
            dense_mult = routing_plan.dense_recall_mult
        if do_hybrid:
            dense_top_k = effective_top_k * dense_mult
        elif routing_plan and routing_plan.route_label in ('keyword', 'long_context'):
            dense_top_k = min(effective_top_k * max(2, dense_mult // 2), 80)
        else:
            dense_top_k = effective_top_k
        with timed_stage('encoder.encode_single'):
            query_vector = self.encoder.encode_single(processed_query, normalize=True)
        with timed_stage('faiss.search'):
            raw_results = self.index.search(query_vector, top_k=dense_top_k)
        if do_hybrid and self._bm25 is not None:
            bm25_results = self._bm25.search(processed_query, top_k=dense_top_k)
            raw_results = self._reciprocal_rank_fusion(dense_results=raw_results, sparse_results=bm25_results, top_k=dense_top_k)
            logger.info('Hybrid fusion: %d dense + %d sparse -> %d merged', len(raw_results), len(bm25_results), len(raw_results))
        results = self._wrap_results(raw_results, score_threshold)
        if filters:
            results = self._apply_filters(results, filters)
        do_rerank = use_reranker if use_reranker is not None else self._reranker is not None
        if do_rerank and self._reranker is not None:
            results = self._reranker.rerank(query_text, results, top_k=effective_top_k)
        else:
            results = results[:effective_top_k]
        route_note = routing_plan.route_label if routing_plan else ''
        logger.info("Query '%s' -> %d results (top_k=%d, threshold=%s, hybrid=%s, rerank=%s, preprocess=%s, route=%s)", query_text[:60], len(results), effective_top_k, score_threshold, do_hybrid, do_rerank, do_preprocess, route_note)
        return results

    def _wrap_results(self, raw_results: List[Dict], score_threshold: Optional[float]=None) -> List[RetrieverResult]:
        results: List[RetrieverResult] = []
        for entry in raw_results:
            if score_threshold is not None and entry['score'] < score_threshold:
                continue
            result = RetrieverResult(chunk_id=entry.get('chunk_id', 'unknown'), doc_id=entry.get('doc_id', 'unknown'), text=entry.get('text', ''), score=entry['score'], metadata={k: v for k, v in entry.items() if k not in ('chunk_id', 'doc_id', 'text', 'score')})
            results.append(result)
        return results

    @staticmethod
    def _reciprocal_rank_fusion(dense_results: List[Dict], sparse_results: List[Dict], top_k: int=20, k: int=60) -> List[Dict]:
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Dict] = {}
        for rank, entry in enumerate(dense_results, start=1):
            key = entry.get('chunk_id', str(rank))
            rrf_scores[key] += 1.0 / (k + rank)
            doc_map[key] = entry
        for rank, entry in enumerate(sparse_results, start=1):
            key = entry.get('chunk_id', f'bm25_{rank}')
            rrf_scores[key] += 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = entry
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        merged: List[Dict] = []
        for key in sorted_keys[:top_k]:
            entry = {**doc_map[key], 'score': rrf_scores[key]}
            merged.append(entry)
        return merged

    @staticmethod
    def _apply_filters(results: List[RetrieverResult], filters: Dict[str, str]) -> List[RetrieverResult]:
        if not filters:
            return results
        filtered: List[RetrieverResult] = []
        for r in results:
            match = True
            for key, value in filters.items():
                actual = getattr(r, key, None)
                if actual is None:
                    actual = r.metadata.get(key)
                if actual is None:
                    match = False
                    break
                if str(value).lower() not in str(actual).lower():
                    match = False
                    break
            if match:
                filtered.append(r)
        logger.info('Metadata filter %s: %d -> %d results', filters, len(results), len(filtered))
        return filtered

    def format_context(self, results: List[RetrieverResult], max_chars: int=3000) -> str:
        parts: List[str] = []
        total = 0
        for i, r in enumerate(results, 1):
            header = f'[Source {i} | score={r.score:.3f} | doc={r.doc_id}]'
            block = f'{header}\n{r.text}\n'
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return '\n'.join(parts)
