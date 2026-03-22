import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from src.defaults import EMBEDDING_MODEL_ID
logger = logging.getLogger(__name__)
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
DEFAULT_MODEL_NAME = EMBEDDING_MODEL_ID
EMBEDDING_DIM = 384

class EmbeddingEncoder:

    def __init__(self, model_name: str=DEFAULT_MODEL_NAME, device: str='cpu', cache_dir: Optional[str]=None):
        if not ST_AVAILABLE:
            raise ImportError('sentence-transformers is required. Install: pip install sentence-transformers')
        self.model_name = model_name
        self.device = device
        logger.info('Loading embedding model: %s on %s', model_name, device)
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        test_emb = self.model.encode(['test'], convert_to_numpy=True)
        self._dim = test_emb.shape[1]
        logger.info('Embedding dimension: %d', self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: List[str], batch_size: int=32, show_progress: bool=False, normalize: bool=True) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True, normalize_embeddings=normalize)
        logger.info('Encoded %d texts -> shape %s', len(texts), embeddings.shape)
        return embeddings.astype(np.float32)

    def encode_single(self, text: str, normalize: bool=True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]

    def save_embeddings(self, embeddings: np.ndarray, output_path: str) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), embeddings)
        logger.info('Saved embeddings (%s) to %s', embeddings.shape, out)
        return out

    @staticmethod
    def load_embeddings(path: str) -> np.ndarray:
        arr = np.load(path)
        logger.info('Loaded embeddings from %s: shape %s', path, arr.shape)
        return arr
if __name__ == '__main__':
    import numpy as np
    import sys
    print('[Encoder Demo] Running sample embedding and similarity test...')
    try:
        encoder = EmbeddingEncoder()
    except ImportError as e:
        print('ERROR:', e)
        sys.exit(1)
    texts = ['The quick brown fox jumps over the lazy dog.', 'A fast brown fox leaps over a sleepy dog.', 'Quantum mechanics describes the behavior of particles at atomic scales.']
    embeddings = encoder.encode(texts)
    print(f'Embeddings shape: {embeddings.shape}')
    for i, emb in enumerate(embeddings):
        print(f'  Text {i}: {emb.shape}, norm={np.linalg.norm(emb):.4f}')

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print('\nCosine similarity matrix:')
    for i in range(len(texts)):
        row = []
        for j in range(len(texts)):
            sim = cosine_sim(embeddings[i], embeddings[j])
            row.append(f'{sim:.3f}')
        print(f'  {i}: {row}')
    print('\n[Done]')
