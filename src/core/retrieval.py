import logging
import re
import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, List, Dict, Optional, Set
import numpy as np
from src.defaults import CROSS_ENCODER_MODEL_ID
from src.core.embedding import EmbeddingEncoder
from src.utils.vectorIndexing import FaissIndex
from src.utils.latencyMonitoring import timed_stage
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from src.retrieval.query_router import QueryPlan
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

class DocumentPassageUnit:
    """
    Represents a retrieved document passage with relevance score.
    """

    def __init__(self, passage_id: str, document_id: str, passage_text: str, relevance_score: float, passage_metadata: Dict):
        self.passage_id = passage_id
        self.document_id = document_id
        self.passage_text = passage_text
        self.relevance_score = relevance_score
        self.passage_metadata = passage_metadata

    def __repr__(self) -> str:
        excerpt = self.passage_text[:75].replace('\n', ' ')
        return f"DocumentPassageUnit(relevance={self.relevance_score:.3f}, text='{excerpt}...')"

    def to_dict(self) -> Dict:
        return {
            'passage_id': self.passage_id,
            'document_id': self.document_id,
            'passage_text': self.passage_text,
            'relevance_score': self.relevance_score,
            'metadata': self.passage_metadata
        }

class TermNormalizer:
    """
    Normalizes and enriches query terms for improved retrieval.
    Corrects common misspellings and suggests semantically related terms.
    """
    
    MISSPELLING_CORRECTIONS: Dict[str, str] = {
        'invoce': 'invoice', 'invioce': 'invoice', 'reciept': 'receipt',
        'recieve': 'receive', 'adress': 'address', 'ammount': 'amount',
        'amoutn': 'amount', 'signiture': 'signature', 'employe': 'employee',
        'documnet': 'document', 'infomation': 'information', 'sumary': 'summary',
        'purchse': 'purchase', 'compnay': 'company', 'accout': 'account',
        'totla': 'total', 'fomr': 'form'
    }
    
    SEMANTIC_EXPANSIONS: Dict[str, List[str]] = {
        'invoice': ['bill', 'receipt', 'statement', 'cost estimate'],
        'employee': ['worker', 'staff', 'personnel', 'team member'],
        'amount': ['total', 'sum', 'value', 'price', 'cost'],
        'address': ['location', 'place', 'destination', 'headquarters'],
        'company': ['organization', 'firm', 'business', 'enterprise'],
        'date': ['day', 'time', 'period', 'timeline'],
        'signature': ['signed', 'authorization', 'approval'],
        'name': ['identifier', 'title', 'label'],
        'purchase': ['acquisition', 'transaction', 'order']
    }

    def normalize_terms(self, query: str, correct_misspellings: bool=True, expand_semantics: bool=True) -> str:
        """
        Apply term normalization and semantic expansion to improve retrieval.
        """
        normalized = query.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.lower()
        
        if correct_misspellings:
            terms = normalized.split()
            corrected_terms = [self.MISSPELLING_CORRECTIONS.get(t, t) for t in terms]
            normalized = ' '.join(corrected_terms)
        
        if expand_semantics:
            semantic_terms = []
            for key, expansion_list in self.SEMANTIC_EXPANSIONS.items():
                if key in normalized:
                    semantic_terms.extend(expansion_list)
            if semantic_terms:
                normalized = normalized + ' ' + ' '.join(semantic_terms)
        
        logger.debug("Normalized query: '%s' -> '%s'", query[:50], normalized[:80])
        return normalized

class KeywordRetriever:
    """
    Keyword-based retrieval using term frequency and inverse document frequency.
    Implements BM25 ranking algorithm for matching queries to documents.
    """

    def __init__(self, k1_parameter: float=1.5, saturation_parameter: float=0.75):
        self.k1_parameter = k1_parameter
        self.saturation_parameter = saturation_parameter
        self._document_tokens: List[List[str]] = []
        self._token_lengths: List[int] = []
        self._mean_length: float = 0.0
        self._total_documents: int = 0
        self._document_frequency: Dict[str, int] = {}
        self._inverse_frequency: Dict[str, float] = {}
        self._document_metadata: List[Dict] = []

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def index_documents(self, documents: List[Dict]) -> None:
        """
        Build keyword index from document collection.
        """
        self._document_metadata = documents
        self._document_tokens = []
        self._document_frequency = defaultdict(int)
        
        for doc in documents:
            tokens = self.tokenize_text(doc.get('text', ''))
            self._document_tokens.append(tokens)
            for term in set(tokens):
                self._document_frequency[term] += 1
        
        self._total_documents = len(documents)
        self._token_lengths = [len(t) for t in self._document_tokens]
        self._mean_length = sum(self._token_lengths) / self._total_documents if self._total_documents else 1.0
        
        # Calculate IDF scores
        for term, doc_freq in self._document_frequency.items():
            self._inverse_frequency[term] = math.log(
                (self._total_documents - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )
        
        logger.info('Indexed %d documents with %d unique terms', self._total_documents, len(self._document_frequency))

    def retrieve_passages(self, query: str, top_k: int=10) -> List[Dict]:
        """
        Retrieve documents ranked by keyword relevance.
        """
        query_tokens = self.tokenize_text(query)
        if not query_tokens or not self._document_tokens:
            return []
        
        relevance_scores = np.zeros(self._total_documents, dtype=np.float64)
        
        for query_term in query_tokens:
            term_idf = self._inverse_frequency.get(query_term, 0.0)
            if term_idf == 0.0:
                continue
            
            for doc_idx, doc_tokens in enumerate(self._document_tokens):
                term_frequency = doc_tokens.count(query_term)
                if term_frequency == 0:
                    continue
                
                doc_length = self._token_lengths[doc_idx]
                bm25_numerator = term_frequency * (self.k1_parameter + 1)
                bm25_denominator = (
                    term_frequency + 
                    self.k1_parameter * (
                        1 - self.saturation_parameter + 
                        self.saturation_parameter * doc_length / self._mean_length
                    )
                )
                relevance_scores[doc_idx] += term_idf * (bm25_numerator / bm25_denominator)
        
        # Return top-k results
        top_k = min(top_k, self._total_documents)
        top_indices = np.argsort(relevance_scores)[::-1][:top_k]
        
        ranked_results: List[Dict] = []
        for idx in top_indices:
            if relevance_scores[idx] <= 0:
                break
            result_entry = {**self._document_metadata[idx], 'score': float(relevance_scores[idx])}
            ranked_results.append(result_entry)
        
        return ranked_results

class RelevanceReranker:
    """
    Re-ranks retrieved passages using neural cross-encoder model.
    Improves ranking based on query-document relevance scores.
    """
    DEFAULT_MODEL = CROSS_ENCODER_MODEL_ID

    def __init__(self, model_identifier: str=DEFAULT_MODEL, device_type: str='cpu'):
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError('sentence-transformers required. Install with: pip install sentence-transformers')
        logger.info('Loading relevance model: %s on %s', model_identifier, device_type)
        self.cross_encoder = CrossEncoder(model_identifier, device=device_type)
        self.model_identifier = model_identifier

    def reorder_by_relevance(self, query: str, passage_results: List['DocumentPassageUnit'], 
                             limit_count: Optional[int]=None) -> List['DocumentPassageUnit']:
        """
        Re-rank passages based on cross-encoder relevance scores.
        """
        if not passage_results:
            return []
        
        # Prepare query-passage pairs for cross-encoder
        qa_pairs = [(query, passage.passage_text) for passage in passage_results]
        
        # Get relevance scores from cross-encoder
        relevance_scores = self.cross_encoder.predict(qa_pairs)
        
        # Sort by relevance score (descending)
        scored_passages = sorted(
            zip(relevance_scores, passage_results),
            key=lambda x: x[0],
            reverse=True
        )
        
        # Rebuild passage objects with new scores
        reranked_passages: List[DocumentPassageUnit] = []
        for new_score, original_passage in scored_passages:
            updated_passage = DocumentPassageUnit(
                passage_id=original_passage.passage_id,
                document_id=original_passage.document_id,
                passage_text=original_passage.passage_text,
                relevance_score=float(new_score),
                passage_metadata={
                    **original_passage.passage_metadata,
                    'original_score': original_passage.relevance_score
                }
            )
            reranked_passages.append(updated_passage)
        
        # Apply limit if specified
        if limit_count is not None:
            reranked_passages = reranked_passages[:limit_count]
        
        logger.info('Reranked %d passages, returned %d', len(passage_results), len(reranked_passages))
        return reranked_passages

class Retriever:

    def __init__(self, encoder: EmbeddingEncoder, index: FaissIndex, enable_bm25: bool=False, 
                 enable_reranker: bool=False, enable_preprocessing: bool=False, 
                 reranker_model: str=RelevanceReranker.DEFAULT_MODEL, reranker_device: str='cpu'):
        self.encoder = encoder
        self.index = index
        
        self._term_normalizer: Optional[TermNormalizer] = None
        if enable_preprocessing:
            self._term_normalizer = TermNormalizer()
            logger.info('Term normalization enabled')
        
        self._keyword_retriever: Optional[KeywordRetriever] = None
        if enable_bm25 and index.metadata:
            self._keyword_retriever = KeywordRetriever()
            self._keyword_retriever.index_documents(index.metadata)
            logger.info('Keyword retrieval enabled (%d documents)', len(index.metadata))
        
        self._relevance_reranker: Optional[RelevanceReranker] = None
        if enable_reranker:
            self._relevance_reranker = RelevanceReranker(
                model_identifier=reranker_model, 
                device_type=reranker_device
            )
            logger.info('Relevance reranking enabled')

    def retrieve_passages(self, query_text: str, top_k: int=5, score_threshold: Optional[float]=None, 
                         filters: Optional[Dict[str, str]]=None, use_hybrid: Optional[bool]=None, 
                         use_reranker: Optional[bool]=None, use_normalization: Optional[bool]=None, 
                         routing_plan: Optional['QueryPlan']=None) -> List[DocumentPassageUnit]:
        """
        Retrieve and rank relevant document passages for the given query.
        """
        if not query_text.strip():
            logger.warning('Empty query - returning no results')
            return []
        
        # Determine effective parameters
        effective_top_k = routing_plan.top_k if routing_plan else top_k
        should_normalize = use_normalization if use_normalization is not None else self._term_normalizer is not None
        
        if routing_plan and routing_plan.use_preprocessing:
            should_normalize = True
        
        # Normalize query terms if enabled
        if should_normalize:
            normalizer = self._term_normalizer or TermNormalizer()
            processed_query = normalizer.normalize_terms(query_text)
            logger.info("Normalized query: '%s' -> '%s'", query_text[:60], processed_query[:80])
        else:
            processed_query = query_text
        
        # Determine retrieval strategy
        use_hybrid_retrieval = use_hybrid if use_hybrid is not None else self._keyword_retriever is not None
        if routing_plan:
            use_hybrid_retrieval = routing_plan.use_hybrid and self._keyword_retriever is not None
        
        # Calculate retrieval depth for hybrid approach
        embedding_multiplier = 3
        if routing_plan:
            embedding_multiplier = routing_plan.dense_recall_mult
        
        if use_hybrid_retrieval:
            embedding_top_k = effective_top_k * embedding_multiplier
        elif routing_plan and routing_plan.route_label in ('keyword', 'long_context'):
            embedding_top_k = min(effective_top_k * max(2, embedding_multiplier // 2), 80)
        else:
            embedding_top_k = effective_top_k
        
        # Retrieve using dense embeddings
        with timed_stage('embedding_retrieval'):
            embedded_query = self.encoder.encode_single(processed_query, normalize=True)
        
        with timed_stage('index_search'):
            dense_passages = self.index.search(embedded_query, top_k=embedding_top_k)
        
        # Hybrid retrieval with keyword matching
        if use_hybrid_retrieval and self._keyword_retriever is not None:
            keyword_passages = self._keyword_retriever.retrieve_passages(processed_query, top_k=embedding_top_k)
            dense_passages = self._combine_retrieval_results(
                dense_results=dense_passages,
                keyword_results=keyword_passages,
                top_k=embedding_top_k
            )
            logger.info('Hybrid: combined %d dense + keyword results', len(dense_passages))
        
        # Wrap and filter results
        passages = self._convert_to_passage_units(dense_passages, score_threshold)
        
        if filters:
            passages = self._apply_metadata_filters(passages, filters)
        
        # Rerank if enabled
        should_rerank = use_reranker if use_reranker is not None else self._relevance_reranker is not None
        if should_rerank and self._relevance_reranker is not None:
            passages = self._relevance_reranker.reorder_by_relevance(
                query_text, passages, 
                limit_count=effective_top_k
            )
        else:
            passages = passages[:effective_top_k]
        
        route_label = routing_plan.route_label if routing_plan else ''
        logger.info(
            "Retrieved %d passages for '%s' (top_k=%d, hybrid=%s, rerank=%s, normalize=%s, route=%s)",
            len(passages), query_text[:50], effective_top_k,
            use_hybrid_retrieval, should_rerank, should_normalize, route_label
        )
        
        return passages

    def _convert_to_passage_units(self, raw_passages: List[Dict], 
                                   score_threshold: Optional[float]=None) -> List[DocumentPassageUnit]:
        """
        Convert raw passage data to DocumentPassageUnit objects.
        """
        passages: List[DocumentPassageUnit] = []
        for entry in raw_passages:
            if score_threshold is not None and entry['score'] < score_threshold:
                continue
            
            passage = DocumentPassageUnit(
                passage_id=entry.get('chunk_id', 'unknown'),
                document_id=entry.get('doc_id', 'unknown'),
                passage_text=entry.get('text', ''),
                relevance_score=entry['score'],
                passage_metadata={
                    k: v for k, v in entry.items() 
                    if k not in ('chunk_id', 'doc_id', 'text', 'score')
                }
            )
            passages.append(passage)
        return passages

    @staticmethod
    def _combine_retrieval_results(dense_results: List[Dict], keyword_results: List[Dict], 
                                   top_k: int=20, normalization_factor: int=60) -> List[Dict]:
        """
        Combine dense and keyword retrieval using reciprocal rank fusion.
        """
        fusion_scores: Dict[str, float] = defaultdict(float)
        passage_map: Dict[str, Dict] = {}
        
        # Add scores from dense retrieval
        for rank, entry in enumerate(dense_results, start=1):
            key = entry.get('chunk_id', str(rank))
            fusion_scores[key] += 1.0 / (normalization_factor + rank)
            passage_map[key] = entry
        
        # Add scores from keyword retrieval
        for rank, entry in enumerate(keyword_results, start=1):
            key = entry.get('chunk_id', f'kw_{rank}')
            fusion_scores[key] += 1.0 / (normalization_factor + rank)
            if key not in passage_map:
                passage_map[key] = entry
        
        # Sort by fusion score
        sorted_keys = sorted(fusion_scores, key=fusion_scores.get, reverse=True)
        
        # Return top-k combined results
        combined: List[Dict] = []
        for key in sorted_keys[:top_k]:
            entry = {**passage_map[key], 'score': fusion_scores[key]}
            combined.append(entry)
        
        return combined

    @staticmethod
    def _apply_metadata_filters(passages: List[DocumentPassageUnit], 
                               filters: Dict[str, str]) -> List[DocumentPassageUnit]:
        """
        Filter passages by metadata constraints.
        """
        if not filters:
            return passages
        
        filtered_passages: List[DocumentPassageUnit] = []
        for passage in passages:
            metadata_matches = True
            for filter_key, filter_value in filters.items():
                attr_value = getattr(passage, filter_key, None)
                if attr_value is None:
                    attr_value = passage.passage_metadata.get(filter_key)
                if attr_value is None or str(attr_value) != filter_value:
                    metadata_matches = False
                    break
            
            if metadata_matches:
                filtered_passages.append(passage)
        
        return filtered_passages
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
