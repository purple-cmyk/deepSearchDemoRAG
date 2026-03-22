import logging
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
from src.defaults import EMBEDDING_MODEL_ID
logger = logging.getLogger(__name__)
EMBEDDING_DIM = 384
DEFAULT_TOKENIZER = EMBEDDING_MODEL_ID

class OVEmbeddingEncoder:

    def __init__(self, model_xml: str='', device: str='CPU', dimension: int=EMBEDDING_DIM, tokenizer_name: str=DEFAULT_TOKENIZER):
        self.model_xml = model_xml
        self.device = device
        self._dim = dimension
        self._compiled_model = None
        self._tokenizer = None
        self._input_names = []
        self._output_name = None
        self._load_tokenizer(tokenizer_name)
        if model_xml:
            self._try_load(model_xml, device)

    def _load_tokenizer(self, tokenizer_name: str) -> None:
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info('Loaded tokenizer: %s', tokenizer_name)
        except ImportError:
            logger.warning('transformers not installed — tokenizer unavailable. Install: pip install transformers')
        except Exception as exc:
            logger.warning("Failed to load tokenizer '%s': %s", tokenizer_name, exc)

    def _try_load(self, model_xml: str, device: str) -> None:
        try:
            try:
                from openvino import Core
            except ImportError:
                from openvino.runtime import Core
            xml_path = Path(model_xml)
            if not xml_path.exists():
                logger.warning('Model file not found: %s', model_xml)
                return
            core = Core()
            model = core.read_model(model=str(xml_path))
            self._compiled_model = core.compile_model(model=model, device_name=device)
            self._input_names = [inp.get_any_name() for inp in self._compiled_model.inputs]
            self._output_name = self._compiled_model.output(0).get_any_name()
            logger.info('Loaded OpenVINO embedding model on %s  inputs=%s  output=%s', device, self._input_names, self._output_name)
        except ImportError:
            logger.warning('openvino not installed — using placeholder mode')
        except Exception as exc:
            logger.warning('Failed to load OV model: %s — using placeholder mode', exc)

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: List[str], batch_size: int=32, show_progress: bool=False, normalize: bool=True) -> np.ndarray:
        if self._compiled_model is None or self._tokenizer is None:
            logger.warning('OVEmbeddingEncoder not fully initialized — returning ZERO vectors for %d texts. Ensure model_xml path is correct and transformers is installed.', len(texts))
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        batch_embeddings = []
        total = len(texts)
        if show_progress:
            try:
                from tqdm import tqdm
                batch_iterator = tqdm(range(0, total, batch_size), desc='Encoding', unit='batch', total=(total + batch_size - 1) // batch_size)
            except ImportError:
                batch_iterator = range(0, total, batch_size)
        else:
            batch_iterator = range(0, total, batch_size)
        for start in batch_iterator:
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            encoded = self._tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='np')
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            infer_inputs = {self._input_names[0]: input_ids, self._input_names[1]: attention_mask}
            if 'token_type_ids' in self._input_names:
                infer_inputs['token_type_ids'] = np.zeros_like(input_ids)
            outputs = self._compiled_model(infer_inputs)
            token_embeddings = outputs[self._output_name]
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.sum(mask_expanded, axis=1)
            sum_mask = np.clip(sum_mask, a_min=1e-09, a_max=None)
            sentence_embeddings = sum_embeddings / sum_mask
            batch_embeddings.append(sentence_embeddings)
        embeddings = np.concatenate(batch_embeddings, axis=0).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-09, a_max=None)
            embeddings = embeddings / norms
        return embeddings

    def encode_single(self, text: str, normalize: bool=True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]

    def benchmark(self, texts: List[str], batch_size: int=32, n_runs: int=5) -> dict:
        _ = self.encode(texts, batch_size=batch_size)
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = self.encode(texts, batch_size=batch_size)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        times_arr = np.array(times)
        return {'device': self.device, 'n_texts': len(texts), 'batch_size': batch_size, 'n_runs': n_runs, 'mean_ms': float(times_arr.mean() * 1000), 'std_ms': float(times_arr.std() * 1000), 'min_ms': float(times_arr.min() * 1000), 'max_ms': float(times_arr.max() * 1000), 'texts_per_sec': float(len(texts) / times_arr.mean())}

def list_available_devices() -> List[str]:
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core
        return Core().available_devices
    except ImportError:
        logger.warning('openvino not installed -- cannot list devices')
        return []
    except Exception as exc:
        logger.error('Failed to query OpenVINO devices: %s', exc)
        return []
