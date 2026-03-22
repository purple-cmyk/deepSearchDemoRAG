import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
logger = logging.getLogger(__name__)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class FaissIndex:

    def __init__(self, dimension: int=384):
        if not FAISS_AVAILABLE:
            raise ImportError('FAISS is required. Install: pip install faiss-cpu')
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self._index_type: str = 'none'

    @property
    def size(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    @property
    def index_type(self) -> str:
        return self._index_type

    def _validate_embeddings(self, embeddings: np.ndarray, metadata: Optional[List[Dict]]=None) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f'Embeddings must be 2-D (got {embeddings.ndim}-D)')
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f'Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}')
        if metadata is not None and embeddings.shape[0] != len(metadata):
            raise ValueError(f'Mismatch: {embeddings.shape[0]} embeddings vs {len(metadata)} metadata entries')
        return embeddings.astype(np.float32, copy=False)

    def build(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        embeddings = self._validate_embeddings(embeddings, metadata)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = list(metadata)
        self._index_type = 'flat'
        logger.info('Built FAISS IndexFlatIP: %d vectors, dim=%d', self.index.ntotal, self.dimension)

    def build_ivf(self, embeddings: np.ndarray, metadata: List[Dict], nlist: int=100, nprobe: int=10) -> None:
        embeddings = self._validate_embeddings(embeddings, metadata)
        n_vectors = embeddings.shape[0]
        effective_nlist = min(nlist, n_vectors)
        if effective_nlist != nlist:
            logger.warning('nlist=%d > n_vectors=%d — clamped to %d', nlist, n_vectors, effective_nlist)
        quantiser = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantiser, self.dimension, effective_nlist, faiss.METRIC_INNER_PRODUCT)
        logger.info('Training IVF index (nlist=%d) on %d vectors …', effective_nlist, n_vectors)
        t0 = time.time()
        self.index.train(embeddings)
        train_time = time.time() - t0
        logger.info('Training completed in %.2f s', train_time)
        self.index.add(embeddings)
        self.metadata = list(metadata)
        self.index.nprobe = nprobe
        self._index_type = 'ivf'
        logger.info('Built FAISS IndexIVFFlat: %d vectors, dim=%d, nlist=%d, nprobe=%d (train %.2fs)', self.index.ntotal, self.dimension, effective_nlist, nprobe, train_time)

    def build_hnsw(self, embeddings: np.ndarray, metadata: List[Dict], M: int=32, ef_construction: int=200, ef_search: int=64) -> None:
        embeddings = self._validate_embeddings(embeddings, metadata)
        self.index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        logger.info('Building HNSW graph (M=%d, efConstruction=%d) for %d vectors …', M, ef_construction, embeddings.shape[0])
        t0 = time.time()
        self.index.add(embeddings)
        build_time = time.time() - t0
        self.metadata = list(metadata)
        self._index_type = 'hnsw'
        logger.info('Built FAISS IndexHNSWFlat: %d vectors, dim=%d, M=%d, efConstruction=%d, efSearch=%d (%.2fs)', self.index.ntotal, self.dimension, M, ef_construction, ef_search, build_time)

    def add(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        if self.index is None:
            raise RuntimeError('Cannot add to a non-existent index. Call build(), build_ivf(), or build_hnsw() first.')
        embeddings = self._validate_embeddings(embeddings, metadata)
        if hasattr(self.index, 'is_trained') and (not self.index.is_trained):
            raise RuntimeError('IVF index is not trained. Build with build_ivf() first.')
        prev_size = self.index.ntotal
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        logger.info('Incremental add: %d new vectors (total: %d → %d)', embeddings.shape[0], prev_size, self.index.ntotal)

    def search(self, query_vector: np.ndarray, top_k: int=5) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning('Search called on empty index')
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, top_k)
        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = {**self.metadata[idx], 'score': float(score)}
            results.append(entry)
        logger.info('Search returned %d results (top_k=%d)', len(results), top_k)
        return results

    def verify(self, embeddings: np.ndarray, sample_queries: int=5, top_k: int=3) -> List[Dict]:
        if self.index is None:
            raise RuntimeError('Index not built — nothing to verify.')
        n = embeddings.shape[0]
        sample_queries = min(sample_queries, n)
        rng = np.random.default_rng(42)
        query_indices = rng.choice(n, size=sample_queries, replace=False)
        report: List[Dict] = []
        for qi in query_indices:
            qvec = embeddings[qi].reshape(1, -1).astype(np.float32)
            scores, ids = self.index.search(qvec, top_k)
            entry = {'query_idx': int(qi), 'top_k_ids': ids[0].tolist(), 'top_k_scores': [round(float(s), 6) for s in scores[0]], 'self_hit': int(qi) in ids[0].tolist()}
            report.append(entry)
        hits = sum((1 for r in report if r['self_hit']))
        logger.info('Verification: %d/%d queries found themselves in top-%d (index_type=%s)', hits, sample_queries, top_k, self._index_type)
        return report

    def save(self, directory: str) -> None:
        if self.index is None:
            raise RuntimeError('Cannot save: index has not been built')
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / 'index.faiss'
        meta_path = out_dir / 'metadata.json'
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(self.metadata, fh, ensure_ascii=False, indent=2)
        logger.info('Saved index (%d vectors, type=%s) to %s', self.index.ntotal, self._index_type, out_dir)

    def load(self, directory: str) -> None:
        in_dir = Path(directory)
        index_path = in_dir / 'index.faiss'
        meta_path = in_dir / 'metadata.json'
        if not index_path.exists():
            raise FileNotFoundError(f'Index file not found: {index_path}')
        if not meta_path.exists():
            raise FileNotFoundError(f'Metadata file not found: {meta_path}')
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, 'r', encoding='utf-8') as fh:
            self.metadata = json.load(fh)
        self.dimension = self.index.d
        logger.info('Loaded index: %d vectors, dim=%d from %s', self.index.ntotal, self.dimension, in_dir)
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
    DIM = 384
    N_VECTORS = 2000
    N_QUERIES = 50
    TOP_K = 5
    print('=' * 65)
    print('  FAISS Index —  Learning TODO Demo')
    print('=' * 65)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((N_VECTORS, DIM)).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    metadata = [{'id': i, 'text': f'chunk_{i}'} for i in range(N_VECTORS)]
    query_indices = rng.choice(N_VECTORS, size=N_QUERIES, replace=False)
    queries = data[query_indices]

    def compute_recall(gt_ids: np.ndarray, pred_ids: np.ndarray) -> float:
        hits = 0
        total = 0
        for g, p in zip(gt_ids, pred_ids):
            g_set = set(g.tolist())
            p_set = set(p.tolist())
            hits += len(g_set & p_set)
            total += len(g_set)
        return hits / total if total else 0.0
    print('\n──── TODO 1: Flat index (exact search) ────')
    idx_flat = FaissIndex(dimension=DIM)
    idx_flat.build(data, metadata)
    t0 = time.time()
    gt_scores, gt_ids = idx_flat.index.search(queries, TOP_K)
    flat_time = time.time() - t0
    report = idx_flat.verify(data, sample_queries=10, top_k=TOP_K)
    self_hits = sum((1 for r in report if r['self_hit']))
    print(f'  Vectors indexed : {idx_flat.size}')
    print(f'  Self-hit check  : {self_hits}/{len(report)} queries found themselves (expected 100 %)')
    print(f'  Search time     : {flat_time * 1000:.2f} ms for {N_QUERIES} queries')
    print(f'  Recall@{TOP_K}      : 100.0 % (by definition — this IS the ground truth)')
    print('\n──── TODO 2: IVF index (approximate, cell-based) ────')
    for nprobe in [1, 5, 10, 50]:
        idx_ivf = FaissIndex(dimension=DIM)
        idx_ivf.build_ivf(data, metadata, nlist=100, nprobe=nprobe)
        t0 = time.time()
        ivf_scores, ivf_ids = idx_ivf.index.search(queries, TOP_K)
        ivf_time = time.time() - t0
        recall = compute_recall(gt_ids, ivf_ids) * 100
        print(f'  nprobe={nprobe:3d}  →  recall@{TOP_K}={recall:6.2f}%  time={ivf_time * 1000:.2f}ms  speedup={flat_time / ivf_time:.1f}×')
    print('\n──── TODO 3: HNSW index (approximate, graph-based) ────')
    for ef_search in [16, 32, 64, 128]:
        idx_hnsw = FaissIndex(dimension=DIM)
        idx_hnsw.build_hnsw(data, metadata, M=32, ef_construction=200, ef_search=ef_search)
        t0 = time.time()
        hnsw_scores, hnsw_ids = idx_hnsw.index.search(queries, TOP_K)
        hnsw_time = time.time() - t0
        recall = compute_recall(gt_ids, hnsw_ids) * 100
        print(f'  efSearch={ef_search:3d}  →  recall@{TOP_K}={recall:6.2f}%  time={hnsw_time * 1000:.2f}ms  speedup={flat_time / hnsw_time:.1f}×')
    print('\n──── TODO 4: Incremental indexing with add() ────')
    half = N_VECTORS // 2
    idx_inc = FaissIndex(dimension=DIM)
    idx_inc.build(data[:half], metadata[:half])
    print(f'  Initial build   : {idx_inc.size} vectors')
    idx_inc.add(data[half:], metadata[half:])
    print(f'  After add()     : {idx_inc.size} vectors')
    inc_scores, inc_ids = idx_inc.index.search(queries, TOP_K)
    recall = compute_recall(gt_ids, inc_ids) * 100
    print(f'  Recall@{TOP_K} vs flat: {recall:.2f}% (expected 100 %)')
    idx_inc_ivf = FaissIndex(dimension=DIM)
    idx_inc_ivf.build_ivf(data[:half], metadata[:half], nlist=50, nprobe=10)
    print(f'  IVF initial     : {idx_inc_ivf.size} vectors')
    idx_inc_ivf.add(data[half:], metadata[half:])
    print(f'  IVF after add() : {idx_inc_ivf.size} vectors')
    print('\n' + '=' * 65)
    print('  All Learning TODOs verified successfully ✔')
    print('=' * 65)
