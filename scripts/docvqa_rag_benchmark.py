from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'docvqa_bench_results.json'
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
try:
    import psutil as _psutil

    def _rss_mb() -> float:
        return _psutil.Process(os.getpid()).memory_info().rss / 1048576
except ImportError:

    def _rss_mb() -> float:
        return 0.0
HUGGINGFACE_REPO = 'nielsr/docvqa_1200_examples'

def _text_from_words(words: List[str]) -> str:
    return ' '.join((w.strip() for w in words if w.strip()))

def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:12]

class DocVQAExample:
    __slots__ = ('example_id', 'doc_id', 'question', 'answers', 'doc_text', 'chunk_id')

    def __init__(self, example_id: str, doc_id: str, question: str, answers: List[str], doc_text: str):
        self.example_id = example_id
        self.doc_id = doc_id
        self.question = question
        self.answers = [a.strip().lower() for a in answers if a.strip()]
        self.doc_text = doc_text
        self.chunk_id = f'{doc_id}_chunk_0'

def load_docvqa_examples(n: int=60) -> List[DocVQAExample]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        sys.exit(1)
    print(f"  Downloading {n} examples from '{HUGGINGFACE_REPO}' …")
    ds = load_dataset(HUGGINGFACE_REPO, split=f'train[:{n}]')
    seen_hashes: Dict[str, str] = {}
    examples: List[DocVQAExample] = []
    skipped = 0
    for row in ds:
        raw_id = str(row.get('id', f'docvqa_{len(examples):04d}'))
        words = row.get('words') or []
        text = _text_from_words(words)
        if not text.strip():
            skipped += 1
            continue
        h = _content_hash(text)
        if h in seen_hashes:
            doc_id = seen_hashes[h]
        else:
            doc_id = f"dvqa_{raw_id.replace('train_', '').replace('validation_', '').zfill(4)}"
            seen_hashes[h] = doc_id
        query_field = row.get('query', {}) or {}
        if isinstance(query_field, dict):
            question = query_field.get('en') or next(iter(query_field.values()), '')
        else:
            question = str(query_field)
        answers_raw = row.get('answers') or []
        if isinstance(answers_raw, str):
            answers_raw = [answers_raw]
        examples.append(DocVQAExample(example_id=raw_id, doc_id=doc_id, question=question.strip(), answers=answers_raw, doc_text=text))
    print(f'  Loaded {len(examples)} examples ({len(seen_hashes)} unique docs, {skipped} skipped – empty text)')
    return examples

class DocVQAIndex:

    def __init__(self, encoder):
        self.encoder = encoder
        self.faiss_index = None
        self.doc_id_to_idx: Dict[str, int] = {}

    def build(self, examples: List[DocVQAExample]) -> None:
        from src.utils.vectorIndexing import FaissIndex
        seen: Dict[str, DocVQAExample] = {}
        for ex in examples:
            if ex.doc_id not in seen:
                seen[ex.doc_id] = ex
        unique = list(seen.values())
        texts = [ex.doc_text for ex in unique]
        print(f'  Encoding {len(unique)} unique documents …')
        t0 = time.perf_counter()
        embeddings = self.encoder.encode(texts, batch_size=32, show_progress=True, normalize=True)
        enc_ms = (time.perf_counter() - t0) * 1000
        print(f'  Encoding done in {enc_ms:.0f} ms  shape={embeddings.shape}')
        metadata = []
        for i, ex in enumerate(unique):
            self.doc_id_to_idx[ex.doc_id] = i
            metadata.append({'chunk_id': ex.chunk_id, 'doc_id': ex.doc_id, 'text': ex.doc_text, 'metadata': {'dataset': 'docvqa', 'file_type': 'image', 'modality': 'image', 'created_year': None, 'source_directory': None, 'file_name': ex.doc_id}})
        self.faiss_index = FaissIndex(dimension=self.encoder.dimension)
        self.faiss_index.build(embeddings, metadata)
        print(f'  FAISS index built  ({self.faiss_index.size} vectors)')

def _get_doc_id(r: Any) -> str:
    if isinstance(r, dict):
        return r.get('doc_id', '')
    return getattr(r, 'doc_id', '')

def recall_at_k(expected_doc_id: str, results: List[Any], k: int) -> bool:
    for r in results[:k]:
        if _get_doc_id(r) == expected_doc_id:
            return True
    return False

def answer_in_context(answers: List[str], context_text: str) -> bool:
    ctx_lower = context_text.lower()
    return any((ans.lower() in ctx_lower for ans in answers if ans))

def token_f1(prediction: str, ground_truth: str) -> float:
    import re

    def _tokenise(s: str) -> List[str]:
        return re.findall('[a-z0-9]+', s.lower())
    pred_toks = _tokenise(prediction)
    gt_toks = _tokenise(ground_truth)
    if not pred_toks or not gt_toks:
        return float(pred_toks == gt_toks)
    from collections import Counter
    common = Counter(pred_toks) & Counter(gt_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pred_toks)
    r = n_common / len(gt_toks)
    return 2 * p * r / (p + r)

def best_f1_against_answers(retrieved_text: str, answers: List[str]) -> float:
    if not answers:
        return 0.0
    return max((token_f1(retrieved_text, a) for a in answers))
_BOLD = '\x1b[1m'
_DIM = '\x1b[2m'
_GREEN = '\x1b[32m'
_CYAN = '\x1b[36m'
_RESET = '\x1b[0m'

class DocVQABenchmark:
    TOP_K_LEVELS = (1, 3, 5)

    def __init__(self, n_docs: int=60, top_k: int=5, verbose: bool=False):
        self.n_docs = n_docs
        self.top_k = top_k
        self.verbose = verbose
        self._encoder = None
        self._dv_index = None
        self._retriever = None
        self._staged = None
        self._examples: List[DocVQAExample] = []

    def setup(self) -> bool:
        print('=' * 70)
        print(f'  {_BOLD}DocVQA RAG Accuracy & Latency Benchmark{_RESET}')
        print('=' * 70)
        print()
        print('  [1/4] Loading embedding encoder …')
        try:
            from src.embeddings.encoder import EmbeddingEncoder
            self._encoder = EmbeddingEncoder(device='cpu')
            print(f'    Encoder ready  (dim={self._encoder.dimension})')
        except Exception as e:
            print(f'    ERROR: {e}')
            return False
        print(f'\n  [2/4] Loading {self.n_docs} DocVQA examples …')
        self._examples = load_docvqa_examples(self.n_docs)
        if not self._examples:
            print('    ERROR: no examples loaded')
            return False
        print('\n  [3/4] Building document index …')
        self._dv_index = DocVQAIndex(self._encoder)
        self._dv_index.build(self._examples)
        print('\n  [4/4] Creating retrievers …')
        from src.retrieval.retriever import Retriever
        from src.retrieval.metadata_filter import MetadataStore, StagedRetriever, QueryMetadataParser
        self._retriever = Retriever(encoder=self._encoder, index=self._dv_index.faiss_index)
        meta_store = MetadataStore(self._dv_index.faiss_index.metadata)
        self._staged = StagedRetriever(encoder=self._encoder, index=self._dv_index.faiss_index, metadata_store=meta_store, parser=QueryMetadataParser())
        print(f'    Both retrievers ready  ({self._dv_index.faiss_index.size} vectors in index)')
        print()
        return True

    def _warmup(self) -> None:
        dummy = 'What is the date on this document?'
        try:
            self._retriever.query(dummy, top_k=self.top_k)
            self._staged.query(dummy, top_k=self.top_k)
        except Exception:
            pass

    def _eval_original(self, ex: DocVQAExample) -> Dict[str, Any]:
        mem_before = _rss_mb()
        t0 = time.perf_counter()
        results = self._retriever.query(ex.question, top_k=self.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        mem_after = _rss_mb()
        context = self._retriever.format_context(results, max_chars=4000)
        r1 = recall_at_k(ex.doc_id, results, 1)
        r3 = recall_at_k(ex.doc_id, results, 3)
        r5 = recall_at_k(ex.doc_id, results, 5)
        aic = answer_in_context(ex.answers, context)
        return {'latency_ms': latency_ms, 'recall_at_1': r1, 'recall_at_3': r3, 'recall_at_5': r5, 'answer_in_context': aic, 'vector_comparisons': self._dv_index.faiss_index.size, 'peak_memory_mb': max(mem_before, mem_after), 'result_doc_ids': [_get_doc_id(r) for r in results], 'context_snippet': context[:200]}

    def _eval_staged(self, ex: DocVQAExample) -> Dict[str, Any]:
        mem_before = _rss_mb()
        t0 = time.perf_counter()
        raw_results, stats = self._staged.query(ex.question, top_k=self.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        mem_after = _rss_mb()
        results = self._retriever._wrap_results(raw_results)
        context = self._retriever.format_context(results, max_chars=4000)
        r1 = recall_at_k(ex.doc_id, results, 1)
        r3 = recall_at_k(ex.doc_id, results, 3)
        r5 = recall_at_k(ex.doc_id, results, 5)
        aic = answer_in_context(ex.answers, context)
        candidates = stats.get('candidates_after_filter', self._dv_index.faiss_index.size)
        if stats.get('fallback_to_full'):
            candidates = self._dv_index.faiss_index.size
        return {'latency_ms': latency_ms, 'recall_at_1': r1, 'recall_at_3': r3, 'recall_at_5': r5, 'answer_in_context': aic, 'vector_comparisons': candidates, 'peak_memory_mb': max(mem_before, mem_after), 'result_doc_ids': [_get_doc_id(r) for r in results], 'context_snippet': context[:200], 'constraints': stats.get('constraints_detected', {}), 'used_prefilter': stats.get('used_prefilter', False), 'fallback_to_full': stats.get('fallback_to_full', False), 'filter_latency_ms': stats.get('filter_latency_ms', 0.0), 'encoding_latency_ms': stats.get('encoding_latency_ms', 0.0), 'search_latency_ms': stats.get('search_latency_ms', 0.0)}

    def run(self) -> List[Dict]:
        print(f'  Warming up encoder …')
        self._warmup()
        print(f'  Warmup done.\n')
        n = len(self._examples)
        total_vecs = self._dv_index.faiss_index.size
        print(f'  Evaluating {n} questions against {total_vecs} indexed docs (top-k={self.top_k}) …\n')
        records = []
        for i, ex in enumerate(self._examples, 1):
            if self.verbose:
                print(f'  [{i:3d}/{n}] Q: {ex.question[:70]}')
                print(f'         expected_doc={ex.doc_id}  answers={ex.answers[:2]}')
            orig = self._eval_original(ex)
            staged = self._eval_staged(ex)
            prec_overlap = len(set(orig['result_doc_ids']) & set(staged['result_doc_ids'])) / max(len(staged['result_doc_ids']), 1)
            if self.verbose:
                r1_match = '✔' if orig['recall_at_1'] else '✘'
                s1_match = '✔' if staged['recall_at_1'] else '✘'
                print(f"         Orig  {orig['latency_ms']:6.1f}ms R@1={r1_match}  AiC={('Y' if orig['answer_in_context'] else 'N')}")
                print(f"         Staged{staged['latency_ms']:5.1f}ms R@1={s1_match}  AiC={('Y' if staged['answer_in_context'] else 'N')}  filter={staged.get('constraints', {})}")
                print()
            elif i % 10 == 0 or i == n:
                print(f'    Progress: {i}/{n}')
            records.append({'example_id': ex.example_id, 'doc_id': ex.doc_id, 'question': ex.question, 'answers': ex.answers, 'original': orig, 'staged': staged, 'precision_overlap': prec_overlap})
        return records

    @staticmethod
    def _pct(hits: int, total: int) -> str:
        if total == 0:
            return '  N/A'
        return f'{hits / total * 100:5.1f}%'

    def report(self, records: List[Dict]) -> None:
        n = len(records)
        if n == 0:
            print('No records to report.')
            return

        def avg(key: str, mode: str) -> float:
            return sum((r[mode][key] for r in records)) / n

        def hits(key: str, mode: str) -> int:
            return sum((1 for r in records if r[mode][key]))
        orig_lat = avg('latency_ms', 'original')
        staged_lat = avg('latency_ms', 'staged')
        lat_delta = orig_lat - staged_lat
        lat_pct = lat_delta / orig_lat * 100 if orig_lat > 0 else 0

        def recall_row(k: int) -> Tuple[str, str, str]:
            key = f'recall_at_{k}'
            oh = hits(key, 'original')
            sh = hits(key, 'staged')
            return (self._pct(oh, n), self._pct(sh, n), '=' if oh == sh else '▲' if sh > oh else '▼')
        r1_o, r1_s, r1_d = recall_row(1)
        r3_o, r3_s, r3_d = recall_row(3)
        r5_o, r5_s, r5_d = recall_row(5)
        aic_o = hits('answer_in_context', 'original')
        aic_s = hits('answer_in_context', 'staged')
        orig_vecs = avg('vector_comparisons', 'original')
        staged_vecs = avg('vector_comparisons', 'staged')
        vec_red = (1 - staged_vecs / orig_vecs) * 100 if orig_vecs > 0 else 0
        prefilter_used = sum((1 for r in records if r['staged'].get('used_prefilter')))
        fallback_count = sum((1 for r in records if r['staged'].get('fallback_to_full')))
        avg_overlap = sum((r['precision_overlap'] for r in records)) / n
        print()
        print('=' * 70)
        print(f'  {_BOLD}DocVQA Benchmark  —  Results Summary{_RESET}')
        print('=' * 70)
        print(f'\n  Dataset : DocVQA ({HUGGINGFACE_REPO})')
        print(f'  Queries : {n}')
        print(f'  Indexed docs: {self._dv_index.faiss_index.size}')
        print(f'  top-k   : {self.top_k}')
        W = 30
        print(f"\n  {'─' * 60}")
        print(f"  {'Metric':<{W}} {'Original':>12} {'Staged':>12}  {'Δ / note':}")
        print(f"  {'─' * 60}")
        print(f"  {'Avg latency (ms)':<{W}} {orig_lat:>12.1f} {staged_lat:>12.1f}  {lat_delta:+.1f} ms ({lat_pct:.1f}% {('faster' if lat_delta > 0 else 'slower')})")
        print(f"  {'Recall@1':<{W}} {r1_o:>12} {r1_s:>12}  {r1_d}")
        print(f"  {'Recall@3':<{W}} {r3_o:>12} {r3_s:>12}  {r3_d}")
        print(f"  {'Recall@5':<{W}} {r5_o:>12} {r5_s:>12}  {r5_d}")
        print(f"  {'Answer-in-context':<{W}} {self._pct(aic_o, n):>12} {self._pct(aic_s, n):>12}")
        print(f"  {'Avg vector comparisons':<{W}} {orig_vecs:>12.0f} {staged_vecs:>12.0f}  {vec_red:.1f}% reduction")
        print(f"  {'Avg result-set overlap':<{W}} {avg_overlap * 100:>11.1f}%")
        print(f"  {'─' * 60}")
        print(f'\n  Pre-filter activated : {prefilter_used}/{n} queries')
        print(f'  Fallback to full     : {fallback_count}/{n} queries')
        if vec_red == 0.0:
            print(f'\n  {_DIM}Note: 0% vector reduction is expected — DocVQA is a')
            print(f'  homogeneous all-image corpus.  The pre-filter returns all')
            print(f"  documents when the constraint is file_type='image'.  Vector")
            print(f'  reduction becomes meaningful on multi-modality corpora.')
            print(f'  The latency advantage of StagedRetriever comes from its')
            print(f'  leaner code path (no cross-encoder / BM25 checks).{_RESET}')
        print(f"\n  {'─' * 70}")
        print(f'  Per-query detail  (sorted by Recall@1 descending, showing 10)')
        print(f"  {'─' * 70}")
        print(f"  {'#':<4} {'Question (truncated)':<40} {'Orig':>6} {'Stgd':>6} {'AiC-O':>6} {'AiC-S':>6}")
        print(f"  {'─' * 70}")
        sorted_recs = sorted(records, key=lambda r: (-r['original']['recall_at_1'], r['original']['latency_ms']))
        for j, row in enumerate(sorted_recs[:10], 1):
            q = row['question'][:38]
            ro1 = '✔' if row['original']['recall_at_1'] else '✘'
            rs1 = '✔' if row['staged']['recall_at_1'] else '✘'
            ao = '✔' if row['original']['answer_in_context'] else '✘'
            as_ = '✔' if row['staged']['answer_in_context'] else '✘'
            print(f'  {j:<4} {q:<40} {ro1:>6} {rs1:>6} {ao:>6} {as_:>6}')
        o_lats = sorted((r['original']['latency_ms'] for r in records))
        s_lats = sorted((r['staged']['latency_ms'] for r in records))

        def _pctile(lst: List[float], p: int) -> float:
            idx = max(0, int(len(lst) * p / 100) - 1)
            return lst[min(idx, len(lst) - 1)]
        print(f"\n  {'─' * 60}")
        print(f'  Latency percentiles (ms)')
        print(f"  {'─' * 60}")
        print(f"  {'Percentile':<20} {'Original':>12} {'Staged':>12}")
        for p in (50, 75, 90, 95, 99):
            op = _pctile(o_lats, p)
            sp = _pctile(s_lats, p)
            print(f"  {'p' + str(p):<20} {op:>12.1f} {sp:>12.1f}")
        print(f"  {'min':<20} {o_lats[0]:>12.1f} {s_lats[0]:>12.1f}")
        print(f"  {'max':<20} {o_lats[-1]:>12.1f} {s_lats[-1]:>12.1f}")
        all_constraints: Dict[str, int] = {}
        for r in records:
            for k in r['staged'].get('constraints', {}):
                all_constraints[k] = all_constraints.get(k, 0) + 1
        if all_constraints:
            print(f'\n  Constraints detected across all queries:')
            for k, v in sorted(all_constraints.items(), key=lambda x: -x[1]):
                print(f'    {k:<20} {v}/{n} queries')
        print()
        print('=' * 70)

def save_results(records: List[Dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = json.dumps(records, indent=2, ensure_ascii=False, default=str)
    RESULTS_PATH.write_text(out, encoding='utf-8')
    print(f'  Results saved → {RESULTS_PATH.relative_to(PROJECT_ROOT)}')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='DocVQA RAG benchmark — original vs metadata-aware retrieval')
    p.add_argument('--n-docs', type=int, default=60, help='Number of DocVQA examples to load (default: 60)')
    p.add_argument('--top-k', type=int, default=5, help='Retrieval top-k (default: 5)')
    p.add_argument('--verbose', action='store_true', help='Print per-query results during evaluation')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    bench = DocVQABenchmark(n_docs=args.n_docs, top_k=args.top_k, verbose=args.verbose)
    if not bench.setup():
        sys.exit(1)
    records = bench.run()
    bench.report(records)
    save_results(records)
if __name__ == '__main__':
    main()
