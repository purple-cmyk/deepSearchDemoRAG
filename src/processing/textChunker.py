import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
logger = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_MIN_CHUNK_LENGTH = 30
_SENTENCE_SPLIT_RE = re.compile('(?<=[.!?])\\s+')
_RECURSIVE_SEPARATORS = ['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ', '']

@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

class TextChunker:

    def __init__(self, chunk_size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP, min_chunk_length: int=DEFAULT_MIN_CHUNK_LENGTH, strategy: str='fixed', tokenizer=None):
        if overlap >= chunk_size:
            raise ValueError(f'Overlap ({overlap}) must be smaller than chunk_size ({chunk_size})')
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length
        self.strategy = strategy
        self.tokenizer = tokenizer
        valid = {'fixed', 'sentence', 'recursive', 'token'}
        if strategy not in valid:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {valid}")
        if strategy == 'token' and tokenizer is None:
            raise ValueError("Token-based chunking requires a tokenizer.  Pass tokenizer= (e.g. AutoTokenizer.from_pretrained('...'))")

    def chunk_text(self, text: str, doc_id: str, base_metadata: Dict=None) -> List[TextChunk]:
        if not text or not text.strip():
            logger.debug('Empty text for doc_id=%s, skipping chunking', doc_id)
            return []
        base_metadata = base_metadata or {}
        if self.strategy == 'sentence':
            chunks = self._chunk_sentence(text, doc_id, base_metadata)
        elif self.strategy == 'recursive':
            chunks = self._chunk_recursive(text, doc_id, base_metadata)
        elif self.strategy == 'token':
            chunks = self._chunk_token(text, doc_id, base_metadata)
        else:
            chunks = self._chunk_fixed(text, doc_id, base_metadata)
        chunks = [c for c in chunks if len(c.text.strip()) >= self.min_chunk_length]
        logger.info('Chunked doc_id=%s into %d chunks (size=%d, overlap=%d)', doc_id, len(chunks), self.chunk_size, self.overlap)
        return chunks

    def _chunk_fixed(self, text: str, doc_id: str, base_metadata: Dict) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunk_meta = {**base_metadata, 'char_start': start, 'char_end': min(end, len(text)), 'strategy': 'fixed'}
            chunk = TextChunk(chunk_id=f'{doc_id}_chunk_{idx:04d}', doc_id=doc_id, text=chunk_text, index=idx, metadata=chunk_meta)
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
            idx += 1
        return chunks

    def _chunk_sentence(self, text: str, doc_id: str, base_metadata: Dict) -> List[TextChunk]:
        sentences = _SENTENCE_SPLIT_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        chunks: List[TextChunk] = []
        idx = 0
        sent_idx = 0
        while sent_idx < len(sentences):
            current_chunk_sents: List[str] = []
            current_len = 0
            while sent_idx < len(sentences):
                sent = sentences[sent_idx]
                added_len = len(sent) + (1 if current_chunk_sents else 0)
                if current_len + added_len > self.chunk_size and current_chunk_sents:
                    break
                current_chunk_sents.append(sent)
                current_len += added_len
                sent_idx += 1
            chunk_text = ' '.join(current_chunk_sents)
            chunk_meta = {**base_metadata, 'sentence_count': len(current_chunk_sents), 'strategy': 'sentence'}
            chunks.append(TextChunk(chunk_id=f'{doc_id}_chunk_{idx:04d}', doc_id=doc_id, text=chunk_text, index=idx, metadata=chunk_meta))
            idx += 1
            if sent_idx < len(sentences) and self.overlap > 0:
                overlap_len = 0
                rewind = 0
                for i in range(len(current_chunk_sents) - 1, -1, -1):
                    s_len = len(current_chunk_sents[i]) + 1
                    if overlap_len + s_len > self.overlap:
                        break
                    overlap_len += s_len
                    rewind += 1
                if rewind > 0:
                    sent_idx -= rewind
        return chunks

    def _chunk_recursive(self, text: str, doc_id: str, base_metadata: Dict, _separators: Optional[List[str]]=None) -> List[TextChunk]:
        separators = _separators if _separators is not None else list(_RECURSIVE_SEPARATORS)
        if not separators:
            return self._chunk_fixed(text, doc_id, base_metadata)
        sep = separators[0]
        remaining_seps = separators[1:]
        if sep == '':
            pieces = list(text)
        else:
            pieces = text.split(sep)
        merged_chunks: List[TextChunk] = []
        current_pieces: List[str] = []
        current_len = 0
        idx = len(merged_chunks)
        for piece in pieces:
            piece_len = len(piece) + (len(sep) if current_pieces else 0)
            if current_len + piece_len <= self.chunk_size:
                current_pieces.append(piece)
                current_len += piece_len
            else:
                if current_pieces:
                    chunk_text = sep.join(current_pieces)
                    merged_chunks.append(TextChunk(chunk_id=f'{doc_id}_chunk_{len(merged_chunks):04d}', doc_id=doc_id, text=chunk_text, index=len(merged_chunks), metadata={**base_metadata, 'strategy': 'recursive'}))
                    current_pieces = []
                    current_len = 0
                if len(piece) > self.chunk_size:
                    sub_chunks = self._chunk_recursive(piece, doc_id, base_metadata, remaining_seps)
                    for sc in sub_chunks:
                        sc.chunk_id = f'{doc_id}_chunk_{len(merged_chunks):04d}'
                        sc.index = len(merged_chunks)
                        merged_chunks.append(sc)
                else:
                    current_pieces = [piece]
                    current_len = len(piece)
        if current_pieces:
            chunk_text = sep.join(current_pieces)
            merged_chunks.append(TextChunk(chunk_id=f'{doc_id}_chunk_{len(merged_chunks):04d}', doc_id=doc_id, text=chunk_text, index=len(merged_chunks), metadata={**base_metadata, 'strategy': 'recursive'}))
        return merged_chunks

    def _chunk_token(self, text: str, doc_id: str, base_metadata: Dict) -> List[TextChunk]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)
        if total_tokens == 0:
            return []
        chunks: List[TextChunk] = []
        start = 0
        idx = 0
        step = self.chunk_size - self.overlap
        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_token_ids = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            chunk_meta = {**base_metadata, 'token_start': start, 'token_end': end, 'token_count': end - start, 'strategy': 'token'}
            chunks.append(TextChunk(chunk_id=f'{doc_id}_chunk_{idx:04d}', doc_id=doc_id, text=chunk_text, index=idx, metadata=chunk_meta))
            start += step
            idx += 1
        return chunks

    def chunk_documents(self, documents) -> List[TextChunk]:
        try:
            from tqdm import tqdm
            iterator = tqdm(documents, desc='Chunking', unit='doc')
        except ImportError:
            iterator = documents
        all_chunks: List[TextChunk] = []
        for doc in iterator:
            doc_chunks = self.chunk_text(text=doc.text, doc_id=doc.doc_id, base_metadata=doc.metadata)
            all_chunks.extend(doc_chunks)
        logger.info('Total chunks across %d documents: %d', len(documents), len(all_chunks))
        return all_chunks

    @staticmethod
    def experiment_chunk_sizes(text: str, doc_id: str='experiment', sizes: Optional[List[int]]=None, overlap: int=DEFAULT_CHUNK_OVERLAP, strategies: Optional[List[str]]=None) -> List[Dict]:
        if sizes is None:
            sizes = [256, 512, 1024]
        if strategies is None:
            strategies = ['fixed', 'sentence', 'recursive']
        results: List[Dict] = []
        for strategy in strategies:
            for size in sizes:
                adj_overlap = min(overlap, size - 1)
                chunker = TextChunker(chunk_size=size, overlap=adj_overlap, strategy=strategy, min_chunk_length=0)
                chunks = chunker.chunk_text(text, doc_id)
                lengths = [len(c.text) for c in chunks]
                row = {'strategy': strategy, 'chunk_size': size, 'overlap': adj_overlap, 'num_chunks': len(chunks), 'avg_chunk_len': sum(lengths) / len(lengths) if lengths else 0, 'min_chunk_len': min(lengths) if lengths else 0, 'max_chunk_len': max(lengths) if lengths else 0}
                results.append(row)
        header = f"{'Strategy':<12} {'Size':>6} {'Overlap':>7} {'Chunks':>7} {'Avg Len':>8} {'Min':>5} {'Max':>5}"
        logger.info('Chunk size experiment on %d chars:', len(text))
        print(f'\n  {header}')
        print(f"  {'─' * len(header)}")
        for r in results:
            print(f"  {r['strategy']:<12} {r['chunk_size']:>6} {r['overlap']:>7} {r['num_chunks']:>7} {r['avg_chunk_len']:>8.0f} {r['min_chunk_len']:>5} {r['max_chunk_len']:>5}")
        print()
        return results
