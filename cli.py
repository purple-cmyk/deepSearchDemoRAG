import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.defaults import CLIP_MODEL_ID, EMBEDDING_MODEL_ID, OV_EMBEDDING_IR_SUBDIR

def _resolve_raw_data_root() -> Path:
    env = os.environ.get('DEEP_SEARCH_RAW_DATA')
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    local = PROJECT_ROOT / 'data' / 'raw'
    if local.is_dir():
        return local
    legacy_wsl = Path('/mnt/d/Openvino-project/data/raw')
    if legacy_wsl.is_dir():
        return legacy_wsl
    legacy_win = Path('D:\\Openvino-project\\data\\raw')
    try:
        if legacy_win.is_dir():
            return legacy_win
    except OSError:
        pass
    return local
RAW_DATA_ROOT = _resolve_raw_data_root()
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
NORMALISED_DIR = PROCESSED_DIR / 'normalised'
OCR_CACHE_DIR = PROCESSED_DIR / 'ocr_cache'
CHUNKS_DIR = PROCESSED_DIR / 'chunks'
EMBEDDINGS_DIR = PROCESSED_DIR / 'embeddings'
INDEX_DIR = PROCESSED_DIR / 'faiss'
SETTINGS = {}
try:
    import yaml
    settings_path = PROJECT_ROOT / 'configs' / 'settings.yaml'
    if settings_path.exists():
        with open(settings_path, 'r', encoding='utf-8') as f:
            SETTINGS = yaml.safe_load(f) or {}
except Exception as e:
    print(f'Warning: Could not load settings.yaml: {e}', file=sys.stderr)
    SETTINGS = {}

def _embedding_model_name() -> str:
    return SETTINGS.get('embeddings', {}).get('model_name', EMBEDDING_MODEL_ID)

def _embedding_device() -> str:
    return SETTINGS.get('embeddings', {}).get('device', 'cpu')

def _effective_cli_settings(args: argparse.Namespace) -> dict:
    from src.runtime.edge_profile import apply_edge_overrides, is_edge_mode
    if getattr(args, 'edge', False):
        return apply_edge_overrides(SETTINGS)
    if is_edge_mode(SETTINGS):
        return apply_edge_overrides(SETTINGS)
    return SETTINGS

def _build_encoder(args: argparse.Namespace):
    from src.utils.performanceFallback import create_embedding_encoder
    from src.utils.latencyMonitoring import get_latency_monitor
    eff = _effective_cli_settings(args)
    enc, meta = create_embedding_encoder(PROJECT_ROOT, eff, getattr(args, 'device', None))
    rt = eff.get('runtime', {})
    mon = get_latency_monitor()
    mon.configure(bool(rt.get('latency_monitoring', True)) and not getattr(args, 'no_latency', False))
    if mon.enabled:
        enc = mon.wrap_encoder(enc)
    return enc, meta, eff
_USE_COLOR = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and (os.environ.get('TERM') != 'dumb')

class _C:
    RESET = '\x1b[0m' if _USE_COLOR else ''
    BOLD = '\x1b[1m' if _USE_COLOR else ''
    DIM = '\x1b[2m' if _USE_COLOR else ''
    GREEN = '\x1b[32m' if _USE_COLOR else ''
    YELLOW = '\x1b[33m' if _USE_COLOR else ''
    BLUE = '\x1b[34m' if _USE_COLOR else ''
    MAGENTA = '\x1b[35m' if _USE_COLOR else ''
    CYAN = '\x1b[36m' if _USE_COLOR else ''
    RED = '\x1b[31m' if _USE_COLOR else ''
    WHITE = '\x1b[97m' if _USE_COLOR else ''
SYM_CHECK = '✔' if _USE_COLOR else '[OK]'
SYM_CROSS = '✘' if _USE_COLOR else '[FAIL]'
SYM_ARROW = '→' if _USE_COLOR else '->'
SYM_DOT = '●' if _USE_COLOR else '*'
SYM_WARN = '⚠' if _USE_COLOR else '[!]'

def _header(title: str, width: int=60) -> None:
    print()
    print(f"{_C.BOLD}{_C.CYAN}{'─' * width}{_C.RESET}")
    print(f'{_C.BOLD}{_C.CYAN}  {title}{_C.RESET}')
    print(f"{_C.BOLD}{_C.CYAN}{'─' * width}{_C.RESET}")

def _step(num: int, total: int, desc: str) -> None:
    print(f'\n  {_C.BOLD}{_C.BLUE}[{num}/{total}]{_C.RESET} {_C.WHITE}{desc}{_C.RESET}')

def _done(msg: str) -> None:
    print(f'      {_C.GREEN}{SYM_CHECK} {msg}{_C.RESET}')

def _warn(msg: str) -> None:
    print(f'      {_C.YELLOW}{SYM_WARN} {msg}{_C.RESET}')

def _info(msg: str) -> None:
    print(f'      {_C.DIM}{msg}{_C.RESET}')

def _error(msg: str) -> None:
    print(f'      {_C.RED}{SYM_CROSS} {msg}{_C.RESET}')

def _summary_table(rows: list[tuple[str, str]], width: int=60) -> None:
    print()
    print(f"  {_C.BOLD}{'─' * (width - 4)}{_C.RESET}")
    for label, value in rows:
        print(f'  {_C.DIM}{label:<22}{_C.RESET}{_C.BOLD}{value}{_C.RESET}')
    print(f"  {_C.BOLD}{'─' * (width - 4)}{_C.RESET}")

def _elapsed(start: float) -> str:
    secs = time.time() - start
    if secs < 60:
        return f'{secs:.1f}s'
    return f'{int(secs // 60)}m {secs % 60:.1f}s'
_NOISY_LOGGERS = ['httpx', 'httpcore', 'urllib3', 'requests', 'huggingface_hub', 'sentence_transformers', 'faiss', 'faiss.loader', 'datasets', 'PIL', 'filelock', 'transformers', 'tokenizers', 'src.ingestion.loader', 'src.ingestion.normalizer', 'src.ingestion.chunker', 'src.ocr', 'src.embeddings.encoder', 'src.index.faiss_index', 'src.retrieval.retriever', 'src.llm']

def setup_logging(verbose: bool=False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
    if not verbose:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

def cmd_ingest(args: argparse.Namespace) -> None:
    from src.processing.documentProcessing import DatasetLoader
    from src.processing.documentNormalizer import DocumentNormalizer
    from src.processing.textChunker import TextChunker
    from src.ocr.tesseract_engine import TesseractEngine
    from src.utils.vectorIndexing import FaissIndex
    dataset = args.dataset
    max_records = getattr(args, 'max_records', 0)
    t0 = time.time()
    _header(f'Ingest Pipeline  {SYM_ARROW}  dataset={_C.MAGENTA}{dataset}{_C.RESET}{_C.BOLD}{_C.CYAN}')
    if max_records:
        _info(f'Max records: {max_records}')
    _step(1, 6, 'Loading dataset')
    loader = DatasetLoader(raw_data_root=str(RAW_DATA_ROOT), image_cache_dir=str(OCR_CACHE_DIR))
    available = loader.list_available_datasets()
    if dataset not in available:
        _error(f"Dataset '{dataset}' not found.")
        _info(f"Available: {', '.join(available)}")
        sys.exit(1)
    raw_docs = loader.load_dataset(dataset, max_records=max_records)
    if not raw_docs:
        _warn('No documents loaded.')
        return
    _done(f'{len(raw_docs)} documents loaded')
    _step(2, 6, 'Running OCR on image documents')
    ocr_count = 0
    ocr_settings = SETTINGS.get('ocr', {})
    engine_name = ocr_settings.get('engine', 'tesseract').lower()
    ocr_engine = None
    try:
        if engine_name == 'paddleocr':
            try:
                from src.ocr.paddle_engine import PaddleOCREngine
                paddle_lang = ocr_settings.get('paddleocr_lang', 'en')
                use_ov = ocr_settings.get('paddleocr_use_openvino', False)
                device = 'CPU'
                ocr_engine = PaddleOCREngine(lang=paddle_lang, use_openvino=use_ov, device=device, confidence_threshold=ocr_settings.get('confidence_threshold', 40))
                _info(f'Using PaddleOCR (lang={paddle_lang}, openvino={use_ov})')
            except ImportError:
                _warn('PaddleOCR not installed, falling back to Tesseract')
                _info('Install with: pip install paddleocr paddlepaddle')
        if ocr_engine is None:
            from src.ocr.tesseract_engine import TesseractEngine
            preprocess = ocr_settings.get('preprocess', True)
            ocr_engine = TesseractEngine(lang=ocr_settings.get('tesseract_lang', 'eng'), preprocess=preprocess, confidence_threshold=ocr_settings.get('confidence_threshold', 40))
            _info(f'Using Tesseract (preprocess={preprocess})')
        images_to_ocr = []
        image_doc_map = {}
        for doc in raw_docs:
            if doc.image_path and (not doc.text):
                images_to_ocr.append(doc.image_path)
                image_doc_map[doc.image_path] = doc
        if images_to_ocr:
            ocr_results = ocr_engine.batch_extract(images_to_ocr)
            for image_path, extracted_text in ocr_results:
                if extracted_text:
                    image_doc_map[image_path].text = extracted_text
                    ocr_count += 1
            _done(f'Extracted text from {ocr_count} images')
        else:
            _info('No images needed OCR (all documents already have text)')
    except ImportError as e:
        _warn(f'OCR skipped — {e}')
        _info('Install Tesseract: pip install pytesseract + sudo apt install tesseract-ocr')
        _info('Or install PaddleOCR: pip install paddleocr paddlepaddle')
    _step(3, 6, 'Normalising documents')
    normalizer = DocumentNormalizer(output_dir=str(NORMALISED_DIR))
    norm_docs = normalizer.normalize(raw_docs)
    normalizer.save(norm_docs)
    _done(f'{len(norm_docs)} documents normalised')
    _step(4, 6, 'Chunking text into passages')
    chunker = TextChunker(chunk_size=512, overlap=64)
    docs_with_text = [d for d in norm_docs if d.text.strip()]
    chunks = chunker.chunk_documents(docs_with_text)
    if not chunks:
        _warn('No text chunks produced — documents may lack extractable text.')
        _info('Ensure OCR is working for image-based datasets.')
        return
    _done(f'{len(chunks)} chunks from {len(docs_with_text)} documents')
    _step(5, 6, 'Generating embeddings')
    encoder, enc_meta, eff = _build_encoder(args)
    if enc_meta.get('backend') == 'openvino':
        _info(f"Encoder: OpenVINO on {enc_meta.get('device')} ({enc_meta.get('model_xml', '')})")
    else:
        _info(f"Encoder: PyTorch ({enc_meta.get('fallback_reason') or 'CPU'})")
    chunk_texts = [c.text for c in chunks]
    bs = int(eff.get('embeddings', {}).get('batch_size', 64))
    embeddings = encoder.encode(chunk_texts, batch_size=bs, show_progress=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as _np
    _np.save(str(EMBEDDINGS_DIR / 'chunk_embeddings.npy'), embeddings)
    _done(f'Shape {embeddings.shape}  ({embeddings.dtype})')
    _step(6, 6, 'Building FAISS index')
    chunk_metadata = [c.to_dict() for c in chunks]
    try:
        from src.retrieval.metadata_filter import enrich_chunk_metadata
        _IMAGE_DATASETS = {'funsd', 'docvqa', 'rvl_cdip'}
        for cm in chunk_metadata:
            meta = cm.get('metadata', {})
            doc_id = cm.get('doc_id', '')
            dataset_name = meta.get('dataset', '')
            stored_file_type = meta.get('file_type') or ''
            stored_modality = meta.get('modality') or ''
            if stored_modality:
                modality = stored_modality
            elif dataset_name in _IMAGE_DATASETS:
                modality = 'image'
            else:
                modality = 'text'
            if stored_file_type:
                file_type = stored_file_type
            elif dataset_name in _IMAGE_DATASETS:
                file_type = 'image'
            else:
                file_type = ''
            enrich_chunk_metadata(cm, file_name=meta.get('file_name', doc_id), file_type=file_type, source_directory=meta.get('source_directory', ''), modality=modality)
    except ImportError:
        pass
    index = FaissIndex(dimension=encoder.dimension)
    index.build(embeddings, chunk_metadata)
    index.save(str(INDEX_DIR))
    _done(f'{index.size} vectors indexed')
    clip_count = 0
    clip_settings = SETTINGS.get('clip', {})
    if clip_settings.get('enabled', False):
        try:
            from src.embeddings.clip_encoder import CLIPEncoder
            from src.utils.vectorIndexing import FaissIndex as CLIPFaissIndex
            _step(7, 7, 'Generating CLIP image embeddings')
            clip_use_ov = clip_settings.get('use_openvino', False)
            clip_model = clip_settings.get('model_name', CLIP_MODEL_ID)
            clip_dim = clip_settings.get('dimension', 512)
            clip_encoder = CLIPEncoder(model_name=clip_model, use_openvino=clip_use_ov)
            if clip_encoder.is_available:
                image_docs = [d for d in norm_docs if d.image_path and Path(d.image_path).exists()]
                if image_docs:
                    clip_embeddings_list = []
                    clip_metadata_list = []
                    for doc in image_docs:
                        emb = clip_encoder.encode_image(doc.image_path)
                        if emb is not None:
                            clip_embeddings_list.append(emb)
                            clip_metadata_list.append({'doc_id': doc.doc_id, 'chunk_id': f'{doc.doc_id}_clip', 'text': doc.text[:200] if doc.text else '', 'image_path': doc.image_path, 'modality': 'image', 'embedding_type': 'clip', 'metadata': doc.metadata})
                            clip_count += 1
                    if clip_embeddings_list:
                        import numpy as _clip_np
                        clip_emb_array = _clip_np.stack(clip_embeddings_list)
                        clip_index = CLIPFaissIndex(dimension=clip_dim)
                        clip_index.build(clip_emb_array, clip_metadata_list)
                        clip_index.save(str(INDEX_DIR / 'clip'))
                        _done(f'{clip_count} CLIP image embeddings indexed')
                    else:
                        _info('No images could be encoded with CLIP')
                else:
                    _info('No image documents found for CLIP embedding')
            else:
                _warn('CLIP encoder not available — skipping image embeddings')
                _info('Install: pip install transformers torch Pillow')
        except ImportError as e:
            _info(f'CLIP embedding skipped — {e}')
        except Exception as e:
            _warn(f'CLIP embedding step failed: {e}')
    summary_rows = [('Documents loaded', str(len(raw_docs))), ('OCR extractions', str(ocr_count)), ('With text', str(len(docs_with_text))), ('Chunks created', str(len(chunks))), ('Vectors indexed', str(index.size))]
    if clip_count:
        summary_rows.append(('CLIP images', str(clip_count)))
    summary_rows.append(('Time elapsed', _elapsed(t0)))
    _summary_table(summary_rows)
    print(f'  {_C.GREEN}{_C.BOLD}{SYM_CHECK} Ingestion complete{_C.RESET}\n')

def cmd_search(args: argparse.Namespace) -> None:
    from src.utils.vectorIndexing import FaissIndex
    from src.retrieval.query_router import analyze_query
    from src.retrieval.retriever import Retriever
    from src.utils.latencyMonitoring import get_latency_monitor
    query = args.query
    top_k = args.top_k
    _header(f'Search  {SYM_ARROW}  "{_C.WHITE}{query}{_C.RESET}{_C.BOLD}{_C.CYAN}"')
    encoder, enc_meta, eff = _build_encoder(args)
    index = FaissIndex()
    try:
        index.load(str(INDEX_DIR))
    except FileNotFoundError:
        _error('Index not found.')
        _info('Run:  python cli.py ingest --dataset <name>')
        sys.exit(1)
    use_metadata = getattr(args, 'metadata_filtering', False)
    if use_metadata:
        from src.retrieval.metadata_filter import MetadataStore, StagedRetriever, QueryMetadataParser
        metadata_store = MetadataStore(index.metadata)
        staged = StagedRetriever(encoder=encoder, index=index, metadata_store=metadata_store, parser=QueryMetadataParser())
        raw_results, stats = staged.query(query, top_k=top_k)
        retriever = Retriever(encoder=encoder, index=index)
        results = retriever._wrap_results(raw_results)
        if stats.get('constraints_detected'):
            _info(f"Metadata constraints: {stats['constraints_detected']}")
            _info(f"Pre-filter: {stats['candidates_before_filter']} -> {stats['candidates_after_filter']} candidates")
            _info(f"Retrieval latency: {stats['retrieval_latency_ms']:.1f} ms")
    else:
        bm25_on = bool(eff.get('retrieval', {}).get('bm25_enabled', False)) or getattr(args, 'query_routing', False)
        retriever = Retriever(encoder=encoder, index=index, enable_bm25=bm25_on)
        if getattr(args, 'query_routing', False):
            plan = analyze_query(query, top_k, bm25_available=retriever._bm25 is not None)
            _info(f"Query route: {plan.route_label} (top_k={plan.top_k}, hybrid={plan.use_hybrid})")
            results = retriever.query(query, top_k=top_k, routing_plan=plan)
        else:
            results = retriever.query(query, top_k=top_k)
    if get_latency_monitor().enabled:
        _info(f"Latency ms: {get_latency_monitor().summary()}")
    if not results:
        _warn('No results found.')
        return
    use_multimodal = getattr(args, 'multimodal', False)
    if use_multimodal:
        try:
            from src.embeddings.clip_encoder import CLIPEncoder
            from src.utils.vectorIndexing import FaissIndex as CLIPFaissIndex
            from src.retrieval.multimodal_retriever import MultimodalRetriever
            clip_settings = SETTINGS.get('clip', {})
            clip_index_path = INDEX_DIR / 'clip'
            if clip_index_path.exists() and clip_settings.get('enabled', False):
                clip_enc = CLIPEncoder(model_name=clip_settings.get('model_name', CLIP_MODEL_ID), use_openvino=clip_settings.get('use_openvino', False))
                if clip_enc.is_available:
                    clip_idx = CLIPFaissIndex()
                    clip_idx.load(str(clip_index_path))
                    mm = MultimodalRetriever(text_encoder=encoder, text_index=index, clip_encoder=clip_enc, clip_index=clip_idx)
                    mm_results = mm.query(query, top_k=top_k)
                    retriever_temp = Retriever(encoder=encoder, index=index)
                    results = retriever_temp._wrap_results(mm_results)
                    _info('Multimodal CLIP retrieval active')
        except Exception as mm_exc:
            _info(f'Multimodal retrieval not available: {mm_exc}')
    print()
    for i, r in enumerate(results, 1):
        score_color = _C.GREEN if r.score >= 0.5 else _C.YELLOW if r.score >= 0.3 else _C.RED
        print(f'  {_C.BOLD}{_C.BLUE}#{i}{_C.RESET}  {score_color}score {r.score:.4f}{_C.RESET}  {_C.DIM}doc={r.doc_id}  chunk={r.chunk_id}{_C.RESET}')
        preview = r.text[:300].replace('\n', ' ').strip()
        print(f'     {preview}')
        print()
    _info(f'Showing {len(results)} result(s) for top_k={top_k}')
    if use_metadata:
        _info('Mode: metadata-aware staged retrieval')
    if use_multimodal:
        _info('Mode: multimodal (text + CLIP)')
    print()

def cmd_ask(args: argparse.Namespace) -> None:
    from src.utils.vectorIndexing import FaissIndex
    from src.llm.ollama_client import OllamaClient
    from src.retrieval.query_router import analyze_query
    from src.retrieval.retriever import Retriever
    from src.utils.latencyMonitoring import get_latency_monitor, timed_stage
    question = args.question
    top_k = args.top_k
    _header(f'Ask  {SYM_ARROW}  "{_C.WHITE}{question}{_C.RESET}{_C.BOLD}{_C.CYAN}"')
    encoder, enc_meta, eff = _build_encoder(args)
    index = FaissIndex()
    try:
        index.load(str(INDEX_DIR))
    except FileNotFoundError:
        _error('Index not found.')
        _info('Run:  python cli.py ingest --dataset <name>')
        sys.exit(1)
    _step(1, 2, 'Retrieving relevant context')
    use_metadata = getattr(args, 'metadata_filtering', False)
    if use_metadata:
        from src.retrieval.metadata_filter import MetadataStore, StagedRetriever, QueryMetadataParser
        metadata_store = MetadataStore(index.metadata)
        staged = StagedRetriever(encoder=encoder, index=index, metadata_store=metadata_store, parser=QueryMetadataParser())
        raw_results, stats = staged.query(question, top_k=top_k)
        retriever = Retriever(encoder=encoder, index=index)
        results = retriever._wrap_results(raw_results)
        context = retriever.format_context(results, max_chars=3000)
        if stats.get('constraints_detected'):
            _info(f"Metadata constraints: {stats['constraints_detected']}")
            _info(f"Pre-filter: {stats['candidates_before_filter']} -> {stats['candidates_after_filter']} candidates")
    else:
        bm25_on = bool(eff.get('retrieval', {}).get('bm25_enabled', False)) or getattr(args, 'query_routing', False)
        retriever = Retriever(encoder=encoder, index=index, enable_bm25=bm25_on)
        if getattr(args, 'query_routing', False):
            plan = analyze_query(question, top_k, bm25_available=retriever._bm25 is not None)
            _info(f"Query route: {plan.route_label} (top_k={plan.top_k})")
            results = retriever.query(question, top_k=top_k, routing_plan=plan)
        else:
            results = retriever.query(question, top_k=top_k)
        context = retriever.format_context(results, max_chars=3000)
    if get_latency_monitor().enabled:
        _info(f"Retrieval latency ms: {get_latency_monitor().summary()}")
    use_multimodal = getattr(args, 'multimodal', False)
    if use_multimodal:
        try:
            from src.embeddings.clip_encoder import CLIPEncoder
            from src.utils.vectorIndexing import FaissIndex as CLIPFaissIndex
            from src.retrieval.multimodal_retriever import MultimodalRetriever
            clip_settings = SETTINGS.get('clip', {})
            clip_index_path = INDEX_DIR / 'clip'
            if clip_index_path.exists() and clip_settings.get('enabled', False):
                clip_enc = CLIPEncoder(model_name=clip_settings.get('model_name', CLIP_MODEL_ID), use_openvino=clip_settings.get('use_openvino', False))
                if clip_enc.is_available:
                    clip_idx = CLIPFaissIndex()
                    clip_idx.load(str(clip_index_path))
                    mm = MultimodalRetriever(text_encoder=encoder, text_index=index, clip_encoder=clip_enc, clip_index=clip_idx)
                    mm_results = mm.query(question, top_k=top_k)
                    results = retriever._wrap_results(mm_results)
                    context = retriever.format_context(results, max_chars=3000)
                    _info('Multimodal CLIP retrieval active')
        except Exception as mm_exc:
            _info(f'Multimodal retrieval not available: {mm_exc}')
    if not results:
        _warn('No relevant context found in the knowledge base.')
        return
    _done(f'{len(results)} chunks retrieved')
    _step(2, 2, 'Generating answer with LLM')
    llm_settings = SETTINGS.get('llm', {})
    provider = llm_settings.get('provider', 'ollama').lower()
    llm = None
    if provider == 'openvino':
        try:
            from src.llm.openvino_llm import OVLLMClient
            from src.openvino.device_manager import DeviceManager
            dm = DeviceManager()
            ov_settings = SETTINGS.get('openvino', {})
            llm_model_dir = ov_settings.get('llm_model_dir', '')
            device = ov_settings.get('device', 'CPU')
            if llm_model_dir and Path(llm_model_dir).exists():
                llm = OVLLMClient(model_dir=llm_model_dir, device=device)
                if llm.is_available():
                    _info(f'Using OpenVINO LLM from {llm_model_dir} on {device}')
                else:
                    _warn('OpenVINO LLM model failed to load, falling back to Ollama')
                    llm = None
            else:
                _warn(f"OpenVINO LLM model not found at '{llm_model_dir}', falling back to Ollama")
                _info('Convert a model first: optimum-cli export openvino --model mistralai/Mistral-7B-Instruct-v0.2 --weight-format int4 models/ov/mistral-7b-instruct')
        except ImportError as e:
            _warn(f'OpenVINO LLM not available ({e}), falling back to Ollama')
    if llm is None:
        from src.llm.ollama_client import OllamaClient
        llm = OllamaClient()
        if llm.is_available():
            _info('Using Ollama LLM')
    if not llm.is_available():
        _warn('No LLM available — showing retrieved context only.')
        _info('For Ollama: ollama serve && ollama pull mistral')
        _info("For OpenVINO: Set llm.provider='openvino' in settings.yaml and convert a model")
        print(f'\n{context}')
        return
    _info('Thinking...')
    use_stream = not getattr(args, 'no_stream', False)
    template = getattr(args, 'template', 'lean')
    if use_stream and hasattr(llm, 'generate_stream'):
        print(f'\n  {_C.BOLD}{_C.WHITE}Q: {question}{_C.RESET}')
        print(f'\n  {_C.GREEN}', end='', flush=True)
        answer_chunks = []
        try:
            for chunk in llm.generate_stream(question=question, context=context, template=template):
                print(chunk, end='', flush=True)
                answer_chunks.append(chunk)
        except KeyboardInterrupt:
            pass
        print(_C.RESET)
    else:
        with timed_stage('llm.generate'):
            answer = llm.generate(question=question, context=context, template=template)
        _done('Answer generated')
        print(f'\n  {_C.BOLD}{_C.WHITE}Q: {question}{_C.RESET}')
        print(f'\n  {_C.GREEN}{answer}{_C.RESET}')
    if get_latency_monitor().enabled:
        _info(f"Latency ms: {get_latency_monitor().summary()}")
    print(f'\n  {_C.DIM}Based on {len(results)} retrieved chunks.{_C.RESET}\n')

def cmd_stats(args: argparse.Namespace) -> None:
    _header('Pipeline Statistics')

    def _stat_line(label: str, value: str, ok: bool=True) -> None:
        icon = f'{_C.GREEN}{SYM_CHECK}{_C.RESET}' if ok else f'{_C.YELLOW}{SYM_WARN}{_C.RESET}'
        print(f'  {icon} {_C.DIM}{label:<26}{_C.RESET}{value}')
    print(f'\n  {_C.BOLD}Raw Data{_C.RESET}')
    if RAW_DATA_ROOT.exists():
        datasets = sorted((d.name for d in RAW_DATA_ROOT.iterdir() if d.is_dir()))
        _stat_line('Root', str(RAW_DATA_ROOT))
        _stat_line('Datasets', ', '.join(datasets) if datasets else 'none')
    else:
        _stat_line('Root', f'{RAW_DATA_ROOT}  (not found)', ok=False)
    print(f'\n  {_C.BOLD}Normalised Documents{_C.RESET}')
    norm_file = NORMALISED_DIR / 'documents.jsonl'
    if norm_file.exists():
        with open(norm_file) as f:
            doc_count = sum((1 for line in f if line.strip()))
        _stat_line('File', str(norm_file.relative_to(PROJECT_ROOT)))
        _stat_line('Document count', str(doc_count))
    else:
        _stat_line('Status', 'Not created yet — run ingest', ok=False)
    print(f'\n  {_C.BOLD}Embeddings{_C.RESET}')
    emb_file = EMBEDDINGS_DIR / 'chunk_embeddings.npy'
    if emb_file.exists():
        import numpy as np
        emb = np.load(str(emb_file))
        _stat_line('Shape', str(emb.shape))
        _stat_line('Dtype', str(emb.dtype))
        _stat_line('Size', f'{emb_file.stat().st_size / 1024:.1f} KB')
    else:
        _stat_line('Status', 'Not created yet — run ingest', ok=False)
    print(f'\n  {_C.BOLD}FAISS Index{_C.RESET}')
    index_file = INDEX_DIR / 'index.faiss'
    meta_file = INDEX_DIR / 'metadata.json'
    if index_file.exists():
        _stat_line('Index size', f'{index_file.stat().st_size / 1024:.1f} KB')
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            _stat_line('Indexed chunks', str(len(meta)))
    else:
        _stat_line('Status', 'Not created yet — run ingest', ok=False)
    print(f'\n  {_C.BOLD}LLM Backend{_C.RESET}')
    try:
        from src.llm.ollama_client import OllamaClient
        client = OllamaClient()
        if client.is_available():
            _stat_line('Ollama', f'available  (model: {client.model})')
        else:
            _stat_line('Ollama', f"model '{client.model}' not found or server not running", ok=False)
    except Exception as e:
        _stat_line('Ollama', f'not reachable ({e})', ok=False)
    print()

def cmd_devices(args: argparse.Namespace) -> None:
    _header('OpenVINO Device Discovery')
    print()
    try:
        from src.openvino.device_manager import DeviceManager
        dm = DeviceManager()
        devices = dm.list_devices()
        if devices:
            for d in devices:
                props = dm.device_properties(d)
                name = props.get('FULL_DEVICE_NAME', '')
                arch = props.get('DEVICE_ARCHITECTURE', '')
                opt_req = props.get('OPTIMAL_NUMBER_OF_INFER_REQUESTS', '')
                print(f'  {_C.GREEN}{SYM_DOT}{_C.RESET} {_C.BOLD}{d:8s}{_C.RESET}  {name}')
                if arch:
                    print(f'             {_C.DIM}Architecture: {arch}{_C.RESET}')
                if opt_req:
                    print(f'             {_C.DIM}Optimal parallel requests: {opt_req}{_C.RESET}')
            print()
            selected = dm.select_from_settings()
            ov_enabled = dm.is_openvino_enabled()
            print(f'  {_C.BOLD}Settings:{_C.RESET}')
            print(f'    OpenVINO enabled: {(_C.GREEN if ov_enabled else _C.YELLOW)}{ov_enabled}{_C.RESET}')
            print(f'    Selected device:  {_C.GREEN}{selected}{_C.RESET}')
            model_ir = dm.get_embedding_model_path()
            ir_exists = Path(model_ir).exists() if model_ir else False
            ir_color = _C.GREEN if ir_exists else _C.RED
            print(f"    Embedding IR:     {ir_color}{model_ir} {('(found)' if ir_exists else '(NOT FOUND)')}{_C.RESET}")
            if getattr(args, 'benchmark', False) and model_ir and ir_exists:
                print(f'\n  {_C.BOLD}Benchmarking...{_C.RESET}')
                results = dm.benchmark_devices(model_xml=model_ir)
                for dev, stats in results.items():
                    if 'error' in stats:
                        print(f"    {dev}: {_C.RED}error — {stats['error']}{_C.RESET}")
                    else:
                        print(f"    {_C.GREEN}{SYM_DOT}{_C.RESET} {dev:8s}  mean={stats['mean_ms']:.2f}ms  min={stats['min_ms']:.2f}ms  max={stats['max_ms']:.2f}ms")
        else:
            _warn('No devices found.')
            _info('Install OpenVINO:  pip install openvino')
    except ImportError as e:
        _warn(f'OpenVINO not available: {e}')
    print()

def cmd_ocr_tune(args: argparse.Namespace) -> None:
    from src.processing.documentProcessing import DatasetLoader
    from src.ocr.tesseract_engine import TesseractEngine
    dataset = args.dataset
    sample_idx = args.sample
    show_comparison = args.compare
    _header(f'OCR Tuning  {SYM_ARROW}  dataset={_C.MAGENTA}{dataset}{_C.RESET}{_C.BOLD}{_C.CYAN}  sample={sample_idx}')
    _step(1, 3, 'Loading sample document')
    loader = DatasetLoader(raw_data_root=str(RAW_DATA_ROOT), image_cache_dir=str(OCR_CACHE_DIR))
    available = loader.list_available_datasets()
    if dataset not in available:
        _error(f"Dataset '{dataset}' not found.")
        _info(f"Available: {', '.join(available)}")
        sys.exit(1)
    raw_docs = loader.load_dataset(dataset, max_records=sample_idx + 1)
    if len(raw_docs) <= sample_idx:
        _error(f'Sample index {sample_idx} out of range (loaded {len(raw_docs)} docs)')
        sys.exit(1)
    doc = raw_docs[sample_idx]
    if not doc.image_path:
        _error(f'Sample {sample_idx} has no image.')
        sys.exit(1)
    if not doc.text:
        _warn(f'Sample {sample_idx} has no ground-truth text — comparison skipped.')
        ground_truth_words = []
    else:
        ground_truth_words = doc.text.split()
    _done(f'Loaded: {doc.doc_key}')
    _info(f'Image: {doc.image_path}')
    _info(f'Ground-truth words: {len(ground_truth_words)}')
    if show_comparison and ground_truth_words:
        _step(2, 3, 'Running OCR comparison')
        try:
            engine = TesseractEngine(preprocess=True, confidence_threshold=40, psm=3)
            result = engine.compare_with_ground_truth(doc.image_path, ground_truth_words)
            _done('OCR completed')
            print(f'\n  {_C.BOLD}Results{_C.RESET}')
            print(f"    {_C.DIM}OCR words:{_C.RESET}          {result['ocr_word_count']}")
            print(f"    {_C.DIM}Ground-truth words:{_C.RESET} {result['gt_word_count']}")
            print(f"    {_C.GREEN}Precision:{_C.RESET}         {result['precision']:.1f}%")
            print(f"    {_C.GREEN}Recall:{_C.RESET}            {result['recall']:.1f}%")
            print(f"    {_C.GREEN}Word accuracy:{_C.RESET}     {result['word_accuracy']:.1f}%")
            if result['word_accuracy'] < 70:
                print(f'\n    {_C.YELLOW}{SYM_WARN} Low accuracy — try tuning settings{_C.RESET}')
            print(f'\n  {_C.BOLD}OCR Preview{_C.RESET}')
            preview = result['ocr_text'][:500]
            print(f'    {_C.DIM}{preview}...{_C.RESET}')
        except ImportError as e:
            _error(f'Missing dependency: {e}')
            sys.exit(1)
    if not show_comparison:
        _step(2, 3, 'Testing multiple OCR configurations')
        if not ground_truth_words:
            _warn('No ground truth available — cannot measure accuracy.')
            _info('Tuning requires datasets with text annotations (e.g. FUNSD).')
            return
        try:
            engine = TesseractEngine()
            results = engine.tune_settings(doc.image_path, ground_truth_words)
            _done(f'Tested {len(results)} configurations')
            print(f'\n  {_C.BOLD}Top 5 Configurations{_C.RESET}')
            print(f"  {_C.DIM}{'Rank':<6}{'Recall':<10}{'Precision':<12}{'Words':<8}{'Config'}{_C.RESET}")
            print(f"  {_C.DIM}{'─' * 70}{_C.RESET}")
            for i, r in enumerate(results[:5], 1):
                cfg = r['config']
                preproc = '✓' if cfg.get('preprocess') else '✗'
                conf = cfg.get('confidence_threshold', 40)
                psm = cfg.get('psm', 3)
                recall_color = _C.GREEN if r['recall'] >= 80 else _C.YELLOW if r['recall'] >= 60 else _C.RED
                print(f"  {_C.BOLD}#{i:<5}{_C.RESET}{recall_color}{r['recall']:>5.1f}%{_C.RESET}   {r['precision']:>6.1f}%     {r['ocr_word_count']:>4}    {_C.DIM}preprocess={preproc} conf={conf} psm={psm}{_C.RESET}")
            print()
            best = results[0]
            print(f"  {_C.GREEN}{_C.BOLD}{SYM_CHECK} Best config:{_C.RESET} {_C.BOLD}preprocess={best['config'].get('preprocess')}, confidence_threshold={best['config'].get('confidence_threshold')}, psm={best['config'].get('psm')}{_C.RESET}")
        except ImportError as e:
            _error(f'Missing dependency: {e}')
            sys.exit(1)
    _step(3, 3, 'Done')
    print()

def cmd_ingest_videos(args: argparse.Namespace) -> None:
    from src.video.video_loader import VideoLoader
    from src.video.frame_sampler import FrameSampler
    from src.video.frame_ocr import FrameOCR
    from src.video.video_document_builder import VideoDocumentBuilder
    t0 = time.time()
    video_settings = SETTINGS.get('video', {})
    video_root = getattr(args, 'video_root', None) or os.environ.get('DEEP_SEARCH_VIDEO_ROOT')
    if not video_root:
        video_root = video_settings.get('video_data_root', str(PROJECT_ROOT / 'data' / 'raw' / 'msrvtt'))
    video_root_path = Path(video_root)
    if not video_root_path.is_absolute():
        video_root_path = (PROJECT_ROOT / video_root_path).resolve()
        video_root = str(video_root_path)
    if not video_root_path.exists():
        for alt in (Path('/mnt/d/Openvino-project/data/raw/archive/data/MSRVTT/MSRVTT'), Path('D:\\Openvino-project\\data\\raw\\archive\\data\\MSRVTT\\MSRVTT')):
            try:
                if alt.exists():
                    video_root_path = alt
                    video_root = str(alt)
                    break
            except OSError:
                pass
    frame_interval = float(video_settings.get('frame_interval', 5))
    enable_whisper = video_settings.get('enable_whisper', False)
    whisper_model = video_settings.get('whisper_model_size', 'small')
    whisper_device = video_settings.get('whisper_device', 'cpu')
    enable_ocr = video_settings.get('enable_frame_ocr', True)
    ocr_min_words = int(video_settings.get('ocr_min_words', 3))
    chunk_interval = float(video_settings.get('chunk_interval', 30))
    output_dir = video_settings.get('output_dir', 'data/processed/videos')
    output_dir = str(PROJECT_ROOT / output_dir)
    max_videos = getattr(args, 'max_videos', 0)
    source_name = getattr(args, 'dataset', 'msrvtt')
    _header(f'Video Ingest Pipeline  {SYM_ARROW}  source={_C.MAGENTA}{source_name}{_C.RESET}{_C.BOLD}{_C.CYAN}')
    _info(f'Video root: {video_root}')
    _info(f"Frame interval: {frame_interval}s | Whisper: {('ON' if enable_whisper else 'OFF (caption mode)')} | OCR: {enable_ocr}")
    if max_videos:
        _info(f'Max videos: {max_videos}')
    _step(1, 5, 'Discovering video files')
    loader = VideoLoader(root=video_root, max_files=max_videos)
    videos = loader.discover()
    if not videos:
        _warn(f'No video files found under {video_root}')
        _info('Check the path and ensure .mp4/.avi/.mkv files exist.')
        return
    videos_with_captions = sum((1 for v in videos if v.captions))
    if videos_with_captions:
        _done(f'Found {len(videos)} video(s) — {videos_with_captions} with pre-loaded captions')
    else:
        _done(f'Found {len(videos)} video file(s)')
    _step(2, 5, 'Initialising pipeline components')
    frames_dir = str(Path(output_dir) / 'frames')
    frame_sampler = FrameSampler(output_dir=frames_dir, interval_seconds=frame_interval)
    frame_ocr = FrameOCR(min_word_count=ocr_min_words) if enable_ocr else None
    builder = VideoDocumentBuilder(output_dir=output_dir, chunk_interval=chunk_interval)
    audio_extractor = None
    transcriber = None
    if enable_whisper:
        from src.video.audio_extractor import AudioExtractor
        from src.video.transcription import WhisperTranscriber
        audio_dir = str(Path(output_dir) / 'audio')
        audio_extractor = AudioExtractor(output_dir=audio_dir)
        transcriber = WhisperTranscriber(model_size=whisper_model, device=whisper_device)
    _done('Components ready')
    all_docs = []
    from tqdm import tqdm as _tqdm
    pbar = _tqdm(videos, desc='Processing videos', unit='video')
    for video in pbar:
        pbar.set_postfix_str(Path(video.video_path).name)
        has_captions = bool(video.captions)
        transcript_segments = []
        raw_segments = []
        if enable_whisper and (not has_captions):
            _step(3, 5, f'Transcribing: {Path(video.video_path).name}')
            wav_path = audio_extractor.extract(video.video_path)
            if wav_path:
                _info('Transcribing audio...')
                raw_segments = transcriber.transcribe(wav_path)
                transcript_segments = transcriber.chunk_segments(raw_segments, interval=chunk_interval)
                _done(f'{len(transcript_segments)} transcript chunks')
            else:
                _warn('No audio — skipping transcription')
        _step(3 if has_captions else 4, 5, f'Extracting frames: {Path(video.video_path).name}')
        sampled_frames = frame_sampler.sample(video.video_path, video_id=video.video_id)
        _done(f'{len(sampled_frames)} frames sampled')
        ocr_results = []
        if enable_ocr and sampled_frames:
            _step(4 if has_captions else 5, 5, f'Running OCR on frames: {Path(video.video_path).name}')
            ocr_results = frame_ocr.extract_batch(sampled_frames)
            _done(f'{len(ocr_results)} frames with text')
        elif not enable_ocr:
            _info('Frame OCR disabled in settings')
        duration = frame_sampler.get_video_duration(video.video_path)
        if has_captions:
            _step(5, 5, f'Building document (caption mode): {video.video_id}')
            doc = builder.build_from_captions(video_id=video.video_id, source=source_name, captions=video.captions, video_path=video.video_path, duration=duration, ocr_results=ocr_results, extra_metadata=video.metadata)
        else:
            _step(5, 5, f'Building document (transcript mode): {video.video_id}')
            doc = builder.build(video_id=video.video_id, source=source_name, transcript_segments=raw_segments if raw_segments else [], ocr_results=ocr_results, video_path=video.video_path, duration=duration, extra_metadata=video.metadata)
        all_docs.append(doc)
        _done(f'{len(doc.chunks)} chunks, {len(doc.text)} chars')
    pbar.close()
    if all_docs:
        _info('Saving all video documents...')
        builder.save_batch(all_docs)
        _done(f'Saved {len(all_docs)} video documents to {output_dir}')
    total_chunks = sum((len(d.chunks) for d in all_docs))
    total_chars = sum((len(d.text) for d in all_docs))
    _summary_table([('Videos processed', str(len(all_docs))), ('Total chunks', str(total_chunks)), ('Total characters', str(total_chars)), ('Output directory', output_dir), ('Time elapsed', _elapsed(t0))])
    print(f'  {_C.GREEN}{_C.BOLD}{SYM_CHECK} Video ingestion complete{_C.RESET}\n')

def cmd_benchmark(args: argparse.Namespace) -> None:
    run_embeddings = getattr(args, 'embeddings', False)
    run_llm = getattr(args, 'llm', False)
    run_all = getattr(args, 'all', False)
    if not run_embeddings and (not run_llm):
        run_all = True
    if run_all:
        run_embeddings = True
        run_llm = True
    batch_size = getattr(args, 'batch_size', 16)
    iterations = getattr(args, 'iterations', 20)
    warmup = getattr(args, 'warmup', 3)
    _header('Performance Benchmark')
    ov_settings = SETTINGS.get('openvino', {})
    ov_enabled = ov_settings.get('enabled', False)
    model_xml = ov_settings.get('embedding_model_ir', '')
    llm_model_dir = ov_settings.get('llm_model_dir', '')
    ov_device = ov_settings.get('device', 'CPU')
    llm_settings = SETTINGS.get('llm', {})
    ollama_model = llm_settings.get('model', 'mistral')
    ollama_endpoint = llm_settings.get('endpoint', 'http://localhost:11434')
    if model_xml and (not Path(model_xml).is_absolute()):
        model_xml = str(PROJECT_ROOT / model_xml)
    if llm_model_dir and (not Path(llm_model_dir).is_absolute()):
        llm_model_dir = str(PROJECT_ROOT / llm_model_dir)
    if run_embeddings:
        _step(1 if run_embeddings and run_llm else 1, 2 if run_embeddings and run_llm else 1, 'Running embedding benchmark')
        _info(f'Batch size: {batch_size}  |  Iterations: {iterations}  |  Warmup: {warmup}')
        from src.benchmark.embedding_benchmark import run_embedding_benchmark, print_embedding_results
        run_ov = ov_enabled and bool(model_xml) and Path(model_xml).exists()
        if not run_ov:
            if not ov_enabled:
                _warn('OpenVINO disabled in settings.yaml — embedding benchmark runs PyTorch only.')
            else:
                _warn(f'OpenVINO IR model not found: {model_xml}')
                _info(f'Convert with: optimum-cli export openvino --model {EMBEDDING_MODEL_ID} {OV_EMBEDDING_IR_SUBDIR}/')
        if str(ov_device).upper() == 'ADAPTIVE' and run_ov and model_xml:
            from src.openvino.device_manager import DeviceManager
            dm = DeviceManager(settings_override=SETTINGS)
            ov_device = dm.select_embedding_device(str(model_xml))
        emb_results = run_embedding_benchmark(batch_size=batch_size, n_iterations=iterations, n_warmup=warmup, model_xml=model_xml if run_ov else None, ov_device=ov_device, run_pytorch=True, run_openvino=run_ov)
        print_embedding_results(emb_results)
    if run_llm:
        _step(2 if run_embeddings else 1, 2 if run_embeddings and run_llm else 1, 'Running LLM benchmark')
        from src.benchmark.llm_benchmark import run_llm_benchmark, print_llm_results
        prefer_ov = ov_enabled and bool(llm_model_dir)
        llm_results = run_llm_benchmark(n_iterations=5, n_warmup=1, model_dir=llm_model_dir if prefer_ov else None, ov_device=ov_device, ollama_model=ollama_model, ollama_endpoint=ollama_endpoint, prefer_openvino=prefer_ov)
        print_llm_results(llm_results)
    print(f'  {_C.GREEN}{_C.BOLD}{SYM_CHECK} Benchmark complete{_C.RESET}\n')

def cmd_path_query(args: argparse.Namespace) -> None:
    input_path = args.path
    question = getattr(args, 'ask', None)
    top_k = getattr(args, 'top_k', 5)
    use_metadata = getattr(args, 'metadata_filtering', False)
    t0 = time.time()
    if not question:
        _error('--ask is required when using --path.')
        _info('Usage: python cli.py --path ./docs --ask "Your question"')
        sys.exit(1)
    target = Path(input_path)
    if not target.exists():
        _error(f'Path not found: {target}')
        sys.exit(1)
    eff = _effective_cli_settings(args)
    from src.processing.documentProcessing import DatasetLoader, RawDocument
    from src.processing.documentNormalizer import DocumentNormalizer
    from src.processing.textChunker import TextChunker
    from src.utils.vectorIndexing import FaissIndex
    from src.retrieval.query_router import analyze_query
    from src.retrieval.retriever import Retriever
    _header(f'Path Query  {SYM_ARROW}  {_C.MAGENTA}{target}{_C.RESET}{_C.BOLD}{_C.CYAN}')
    _info(f'Question: {question}')
    total_steps = 7
    _step(1, total_steps, 'Loading documents from path')
    anchor = str(target.parent if target.is_file() else target)
    loader = DatasetLoader(raw_data_root=anchor, image_cache_dir=str(OCR_CACHE_DIR))
    try:
        raw_docs = loader.load_path(str(target))
    except Exception as exc:
        _error(f'Failed to load documents: {exc}')
        sys.exit(1)
    if not raw_docs:
        _warn('No documents found at the provided path.')
        return
    _done(f'{len(raw_docs)} document(s) loaded')
    cached_docs = [d for d in raw_docs if d.metadata.get('is_cached')]
    uncached_docs = [d for d in raw_docs if not d.metadata.get('is_cached')]
    _step(2, total_steps, 'Running OCR on image documents')
    ocr_count = 0
    ocr_settings = SETTINGS.get('ocr', {})
    engine_name = ocr_settings.get('engine', 'tesseract').lower()
    ocr_engine = None
    try:
        if engine_name == 'paddleocr':
            try:
                from src.ocr.paddle_engine import PaddleOCREngine
                paddle_lang = ocr_settings.get('paddleocr_lang', 'en')
                use_ov = ocr_settings.get('paddleocr_use_openvino', False)
                ocr_engine = PaddleOCREngine(lang=paddle_lang, use_openvino=use_ov, device='CPU', confidence_threshold=ocr_settings.get('confidence_threshold', 40))
            except ImportError:
                pass
        if ocr_engine is None:
            from src.ocr.tesseract_engine import TesseractEngine
            ocr_engine = TesseractEngine(lang=ocr_settings.get('tesseract_lang', 'eng'), preprocess=ocr_settings.get('preprocess', True), confidence_threshold=ocr_settings.get('confidence_threshold', 40))
        images_to_ocr = []
        image_doc_map = {}
        for doc in uncached_docs:
            if doc.image_path and (not doc.text):
                images_to_ocr.append(doc.image_path)
                image_doc_map[doc.image_path] = doc
        if images_to_ocr:
            ocr_results = ocr_engine.batch_extract(images_to_ocr)
            for image_path, extracted_text in ocr_results:
                if extracted_text:
                    image_doc_map[image_path].text = extracted_text
                    ocr_count += 1
            _done(f'Extracted text from {ocr_count} images')
        else:
            _info('No images needed OCR')
    except ImportError as e:
        _warn(f'OCR skipped — {e}')
    _step(3, total_steps, 'Normalising documents')
    normalizer = DocumentNormalizer(output_dir=str(NORMALISED_DIR))
    norm_docs = normalizer.normalize(uncached_docs)
    normalizer.save(norm_docs)
    _done(f'{len(norm_docs)} new document(s) normalised')
    _step(4, total_steps, 'Chunking text into passages')
    chunker = TextChunker(chunk_size=512, overlap=64)
    docs_with_text = [d for d in norm_docs if d.text.strip()]
    uncached_chunks = chunker.chunk_documents(docs_with_text)
    from src.processing.textChunker import TextChunk
    all_chunks = list(uncached_chunks)
    for doc in cached_docs:
        c_list = doc.metadata.get('cached_chunks', [])
        for c in c_list:
            all_chunks.append(TextChunk(**c))
    if not all_chunks:
        _warn('No text chunks produced — documents may lack extractable text. Continuing for visual search.')
    else:
        _done(f'{len(all_chunks)} total chunks combined (new and cached)')
    clip_enabled = eff.get('clip', {}).get('enabled', True)
    if clip_enabled:
        _info('Precomputing CLIP embeddings and saving to cache...')
    else:
        _info('CLIP disabled (edge or settings) — skipping frame embeddings')
    uncached_clip_data = {}
    doc_id_to_chunks = {}
    for c in uncached_chunks:
        doc_id_to_chunks.setdefault(c.doc_id, []).append(c.to_dict())
    clip_enc = None
    from src.processing.documentProcessing import CacheManager
    cache_mgr = CacheManager()
    source_to_ndoc = {nd.source: nd for nd in norm_docs}
    for raw_doc in uncached_docs:
        n_doc = source_to_ndoc.get(raw_doc.source)
        ndoc_text = n_doc.text if n_doc else ''
        ndoc_doc_id = n_doc.doc_id if n_doc else raw_doc.doc_key
        fps = list(raw_doc.metadata.get('video_frame_paths', []))
        if raw_doc.image_path:
            fps.append(raw_doc.image_path)
        emb_list = []
        valid_fps = []
        if fps and clip_enabled:
            if clip_enc is None:
                from src.embeddings.clip_encoder import CLIPEncoder
                clip_settings = eff.get('clip', SETTINGS.get('clip', {}))
                use_ov_clip = clip_settings.get('use_openvino', False)
                clip_enc = CLIPEncoder(device='cpu', use_openvino=use_ov_clip)
            unique_fps = list(set(fps))
            _info(f'  Encoding {len(unique_fps)} frame(s) with CLIP for {Path(raw_doc.source).name}...')
            try:
                from tqdm import tqdm as _tqdm
                fp_iter = _tqdm(unique_fps, desc='  CLIP frames', unit='frame', leave=False)
            except ImportError:
                fp_iter = unique_fps
            for fp in fp_iter:
                if Path(fp).exists():
                    emb = clip_enc.encode_image(fp)
                    if emb is not None:
                        emb_list.append(emb.tolist())
                        valid_fps.append(fp)
        clip_data = {'image_paths': valid_fps, 'embeddings': emb_list, 'frame_indices': list(range(len(valid_fps)))}
        uncached_clip_data[raw_doc.source] = clip_data
        safe_meta = {k: v for k, v in raw_doc.metadata.items() if k not in ['cached_chunks', 'cached_clip_data', 'is_cached']}
        cache_data = {'doc_key': raw_doc.doc_key, 'source': raw_doc.source, 'text': ndoc_text, 'image_path': raw_doc.image_path, 'metadata': safe_meta, 'chunks': doc_id_to_chunks.get(ndoc_doc_id, []), 'clip_data': clip_data}
        cache_mgr.save_cache(raw_doc.source, cache_data)
        if valid_fps or ndoc_text:
            _done(f'  Cached {Path(raw_doc.source).name} ({len(valid_fps)} CLIP frames, {len(doc_id_to_chunks.get(ndoc_doc_id, []))} chunks)')
    _step(5, total_steps, 'Generating embeddings')
    encoder, enc_meta, eff = _build_encoder(args)
    if all_chunks:
        chunk_texts = [c.text for c in all_chunks]
        bs = int(eff.get('embeddings', {}).get('batch_size', 64))
        embeddings = encoder.encode(chunk_texts, batch_size=bs, show_progress=True)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        import numpy as _np
        _np.save(str(EMBEDDINGS_DIR / 'chunk_embeddings.npy'), embeddings)
        _done(f'Shape {embeddings.shape}  ({embeddings.dtype})')
    _step(6, total_steps, 'Building FAISS index')
    index = None
    if all_chunks:
        chunk_metadata = [c.to_dict() for c in all_chunks]
        try:
            from src.retrieval.metadata_filter import enrich_chunk_metadata
            for cm in chunk_metadata:
                meta = cm.get('metadata', {})
                doc_id = cm.get('doc_id', '')
                file_type = meta.get('file_type', '')
                modality = meta.get('modality', '')
                if not modality:
                    if file_type in ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'):
                        modality = 'image'
                    elif file_type == 'video':
                        modality = 'video'
                    else:
                        modality = 'text'
                enrich_chunk_metadata(cm, file_name=meta.get('file_name', doc_id), file_type=file_type, source_directory=meta.get('source_directory', ''), modality=modality)
        except ImportError:
            pass
        index = FaissIndex(dimension=encoder.dimension)
        index.build(embeddings, chunk_metadata)
        index.save(str(INDEX_DIR))
        _done(f'{index.size} vectors indexed')
    _step(7, total_steps, 'Retrieving context and generating answer')
    context = ''
    results = []
    if all_chunks and index is not None:
        if use_metadata:
            from src.retrieval.metadata_filter import MetadataStore, StagedRetriever, QueryMetadataParser
            metadata_store = MetadataStore(index.metadata)
            staged = StagedRetriever(encoder=encoder, index=index, metadata_store=metadata_store, parser=QueryMetadataParser())
            raw_results, stats = staged.query(question, top_k=top_k)
            retriever = Retriever(encoder=encoder, index=index)
            results = retriever._wrap_results(raw_results)
            context = retriever.format_context(results, max_chars=3000)
            if stats.get('constraints_detected'):
                _info(f"Metadata constraints: {stats['constraints_detected']}")
                _info(f"Pre-filter: {stats['candidates_before_filter']} -> {stats['candidates_after_filter']} candidates")
        else:
            bm25_on = bool(eff.get('retrieval', {}).get('bm25_enabled', False)) or getattr(args, 'query_routing', False)
            retriever = Retriever(encoder=encoder, index=index, enable_bm25=bm25_on)
            if getattr(args, 'query_routing', False):
                plan = analyze_query(question, top_k, bm25_available=retriever._bm25 is not None)
                _info(f"Query route: {plan.route_label} (top_k={plan.top_k})")
                results = retriever.query(question, top_k=top_k, routing_plan=plan)
            else:
                results = retriever.query(question, top_k=top_k)
            context = retriever.format_context(results, max_chars=3000)
        if not results:
            _warn('No relevant text context found in the knowledge base.')
        else:
            _done(f'{len(results)} chunks retrieved')
    all_clip_paths = []
    all_clip_embs = []
    for raw_doc in uncached_docs:
        cd = uncached_clip_data.get(raw_doc.source, {})
        all_clip_paths.extend(cd.get('image_paths', []))
        all_clip_embs.extend(cd.get('embeddings', []))
    for raw_doc in cached_docs:
        cd = raw_doc.metadata.get('cached_clip_data', {})
        all_clip_paths.extend(cd.get('image_paths', []))
        all_clip_embs.extend(cd.get('embeddings', []))
    clip_visual_context = ''
    if all_clip_paths and all_clip_embs and eff.get('clip', {}).get('enabled', True):
        _info(f'Running CLIP visual search on {len(all_clip_paths)} visual frames...')
        try:
            from src.utils.vectorIndexing import FaissIndex as _FaissIndex
            import numpy as _np
            if clip_enc is None:
                from src.embeddings.clip_encoder import CLIPEncoder
                clip_settings = SETTINGS.get('clip', {})
                use_ov_clip = clip_settings.get('use_openvino', False)
                clip_enc = CLIPEncoder(device='cpu', use_openvino=use_ov_clip)
            frame_matrix = _np.array(all_clip_embs).astype('float32')
            clip_index = _FaissIndex(dimension=clip_enc.dimension)
            clip_meta = [{'text': f'[Visual frame: {Path(fp).name}]', 'source': fp, 'doc_id': Path(fp).stem, 'metadata': {'file_type': 'visual_frame', 'frame_path': fp}} for fp in all_clip_paths]
            clip_index.build(frame_matrix, clip_meta)
            query_clip_emb = clip_enc.encode_text(question)
            if query_clip_emb is not None:
                q_vec = query_clip_emb.reshape(1, -1).astype('float32')
                clip_scores, clip_indices = clip_index.index.search(q_vec, min(top_k, len(all_clip_paths)))
                visual_parts = []
                for score, idx in zip(clip_scores[0], clip_indices[0]):
                    if idx < 0:
                        continue
                    fp = all_clip_paths[idx]
                    visual_parts.append(f'[Frame: {Path(fp).name}  |  visual similarity: {score:.3f}]')
                if visual_parts:
                    clip_visual_context = '\n\n[Visual Context — CLIP matched frames from video/images]\n' + '\n'.join(visual_parts)
                    _done(f'CLIP matched {len(visual_parts)} visually relevant frames')
        except Exception as _clip_exc:
            _warn(f'CLIP visual search skipped: {_clip_exc}')
    full_context = context + clip_visual_context if clip_visual_context else context
    _info('Generating answer with LLM...')
    llm_settings = SETTINGS.get('llm', {})
    provider = llm_settings.get('provider', 'ollama').lower()
    llm = None
    if provider == 'openvino':
        try:
            from src.llm.openvino_llm import OVLLMClient
            ov_settings = SETTINGS.get('openvino', {})
            llm_model_dir = ov_settings.get('llm_model_dir', '')
            device = ov_settings.get('device', 'CPU')
            if llm_model_dir and Path(llm_model_dir).exists():
                llm = OVLLMClient(model_dir=llm_model_dir, device=device)
                if not llm.is_available():
                    llm = None
        except ImportError:
            pass
    if llm is None:
        from src.llm.ollama_client import OllamaClient
        llm = OllamaClient()
    if not llm.is_available():
        _warn('No LLM available — showing retrieved context only.')
        _info('For Ollama: ollama serve && ollama pull mistral')
        print(f'\n{full_context}')
    else:
        answer = llm.generate(question=question, context=full_context)
        _done('Answer generated')
        print(f'\n  {_C.BOLD}{_C.WHITE}Q: {question}{_C.RESET}')
        print(f'\n  {_C.GREEN}{answer}{_C.RESET}')
        extra = ' + CLIP visual frames' if clip_visual_context else ''
        print(f'\n  {_C.DIM}Based on {len(results)} retrieved chunks{extra}.{_C.RESET}')
    _summary_table([('Documents loaded', str(len(raw_docs))), ('OCR extractions', str(ocr_count)), ('Chunks created', str(len(all_chunks))), ('Vectors indexed', str(index.size) if index is not None else '—'), ('CLIP frames', str(len(all_clip_paths)) if all_clip_paths else '—'), ('Time elapsed', _elapsed(t0))])
    print(f'  {_C.GREEN}{_C.BOLD}{SYM_CHECK} Path query complete{_C.RESET}\n')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='deep-search', description='Multimodal RAG-based search assistant for personal knowledge bases.  Powered by OpenVINO for efficient on-device inference.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--edge', action='store_true', help='Edge mode: low memory/CPU (smaller batches, CPU, reduced top-k)')
    parser.add_argument('--no-latency', action='store_true', dest='no_latency', help='Disable inference latency monitoring')
    parser.add_argument('--path', type=str, default=None, help='Path to a local file or directory to ingest and query')
    parser.add_argument('--ask', type=str, default=None, help='Question to answer after ingesting the path (requires --path)')
    parser.add_argument('--metadata-filtering', action='store_true', dest='metadata_filtering', default=False, help='Enable metadata-aware staged retrieval (requires --path)')
    parser.add_argument('--top-k', type=int, default=5, dest='top_k', help='Number of context chunks to retrieve (requires --path, default: 5)')
    parser.add_argument('--query-routing', action='store_true', dest='query_routing', help='Query-aware retrieval for --path mode')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    p_ingest = subparsers.add_parser('ingest', help='Ingest and index documents from a raw dataset')
    p_ingest.add_argument('--dataset', type=str, required=True, help='Dataset name: funsd, docvqa, or rvl_cdip')
    p_ingest.add_argument('--max-records', type=int, default=0, dest='max_records', help='Limit the number of records loaded (0 = all, default: 0)')
    p_ingest.add_argument('--device', type=str, default=None, help='Override OpenVINO device (CPU, GPU, NPU, AUTO, ADAPTIVE). Default: from settings.yaml')
    p_ingest.add_argument('--edge', action='store_true', help='Edge mode (low memory/CPU); can also use: python cli.py --edge ingest ...')
    p_ingest.add_argument('--no-latency', action='store_true', dest='no_latency', help='Disable latency monitoring')
    p_ingest.set_defaults(func=cmd_ingest)
    p_search = subparsers.add_parser('search', help='Semantic search over indexed documents (no LLM)')
    p_search.add_argument('query', type=str, help='Natural-language search query')
    p_search.add_argument('--top-k', type=int, default=5, dest='top_k', help='Number of results to return (default: 5)')
    p_search.add_argument('--device', type=str, default=None, help='Override OpenVINO device (CPU, GPU, NPU, AUTO, ADAPTIVE). Default: from settings.yaml')
    p_search.add_argument('--metadata-filtering', action='store_true', dest='metadata_filtering', help='Enable metadata-aware staged retrieval (pre-filter by year/type/directory)')
    p_search.add_argument('--multimodal', action='store_true', help='Enable multimodal retrieval (text + CLIP image embeddings)')
    p_search.add_argument('--query-routing', action='store_true', dest='query_routing', help='Query-aware retrieval (adjust top-k, preprocessing, optional BM25 hybrid)')
    p_search.add_argument('--edge', action='store_true', help='Edge mode')
    p_search.add_argument('--no-latency', action='store_true', dest='no_latency', help='Disable latency monitoring')
    p_search.set_defaults(func=cmd_search)
    p_ask = subparsers.add_parser('ask', help='Ask a question (RAG: retrieval + LLM generation)')
    p_ask.add_argument('question', type=str, help='Natural-language question')
    p_ask.add_argument('--top-k', type=int, default=5, dest='top_k', help='Number of context chunks to retrieve (default: 5)')
    p_ask.add_argument('--device', type=str, default=None, help='Override OpenVINO device (CPU, GPU, NPU, AUTO, ADAPTIVE). Default: from settings.yaml')
    p_ask.add_argument('--metadata-filtering', action='store_true', dest='metadata_filtering', help='Enable metadata-aware staged retrieval (pre-filter by year/type/directory)')
    p_ask.add_argument('--multimodal', action='store_true', help='Enable multimodal retrieval (text + CLIP image embeddings)')
    p_ask.add_argument('--query-routing', action='store_true', dest='query_routing', help='Query-aware retrieval routing')
    p_ask.add_argument('--edge', action='store_true', help='Edge mode')
    p_ask.add_argument('--no-latency', action='store_true', dest='no_latency', help='Disable latency monitoring')
    p_ask.add_argument('--no-stream', action='store_true', dest='no_stream', help='Disable streaming output; wait for full response before printing')
    p_ask.add_argument('--template', type=str, default='lean', choices=['lean', 'default', 'cited', 'concise'], help='Prompt template (default: lean — minimal tokens for fastest streaming)')
    p_ask.set_defaults(func=cmd_ask)
    p_stats = subparsers.add_parser('stats', help='Show pipeline and index statistics')
    p_stats.set_defaults(func=cmd_stats)
    p_devices = subparsers.add_parser('devices', help='List available OpenVINO hardware devices')
    p_devices.add_argument('--benchmark', action='store_true', help='Run inference benchmark on all available devices')
    p_devices.set_defaults(func=cmd_devices)
    p_ocr = subparsers.add_parser('ocr-tune', help='Test and tune OCR settings on sample images')
    p_ocr.add_argument('--dataset', type=str, required=True, help='Dataset name (must have ground-truth text, e.g. funsd)')
    p_ocr.add_argument('--sample', type=int, default=0, help='Sample index to test (default: 0)')
    p_ocr.add_argument('--compare', action='store_true', help='Show single OCR comparison instead of tuning multiple configs')
    p_ocr.set_defaults(func=cmd_ocr_tune)
    p_video = subparsers.add_parser('ingest-videos', help='Ingest video files: extract audio, transcribe, OCR frames, build JSON')
    p_video.add_argument('--dataset', type=str, default='msrvtt', help='Video dataset source name (default: msrvtt)')
    p_video.add_argument('--video-root', type=str, default=None, dest='video_root', help='Override path to video dataset directory (default: from settings.yaml)')
    p_video.add_argument('--max-videos', type=int, default=0, dest='max_videos', help='Limit the number of videos to process (0 = all, default: 0)')
    p_video.set_defaults(func=cmd_ingest_videos)
    p_bench = subparsers.add_parser('benchmark', help='Run performance benchmarks (embeddings and/or LLM)')
    p_bench.add_argument('--embeddings', action='store_true', help='Benchmark embedding inference (PyTorch vs OpenVINO)')
    p_bench.add_argument('--llm', action='store_true', help='Benchmark LLM inference (OpenVINO or Ollama)')
    p_bench.add_argument('--all', action='store_true', help='Run all benchmarks (default when no flag is given)')
    p_bench.add_argument('--batch-size', type=int, default=16, dest='batch_size', help='Batch size for embedding benchmark (default: 16)')
    p_bench.add_argument('--iterations', type=int, default=20, dest='iterations', help='Number of timed iterations (default: 20)')
    p_bench.add_argument('--warmup', type=int, default=3, dest='warmup', help='Number of warmup iterations excluded from timing (default: 3)')
    p_bench.set_defaults(func=cmd_benchmark)
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=getattr(args, 'verbose', False))
    if getattr(args, 'path', None):
        try:
            cmd_path_query(args)
        except KeyboardInterrupt:
            print(f'\n  {_C.YELLOW}Interrupted by user.{_C.RESET}')
            sys.exit(130)
        except Exception as exc:
            if getattr(args, 'verbose', False):
                logging.error('Path query failed: %s', exc, exc_info=True)
            else:
                print(f'\n  {_C.RED}{SYM_CROSS} Error: {exc}{_C.RESET}')
                print(f'  {_C.DIM}Run with --verbose for full traceback.{_C.RESET}')
            sys.exit(1)
        return
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print(f'\n  {_C.YELLOW}Interrupted by user.{_C.RESET}')
        sys.exit(130)
    except ImportError as exc:
        print(f'\n  {_C.RED}{SYM_CROSS} Missing dependency: {exc}{_C.RESET}')
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f'\n  {_C.RED}{SYM_CROSS} File not found: {exc}{_C.RESET}')
        sys.exit(1)
    except Exception as exc:
        if getattr(args, 'verbose', False):
            logging.error('Command failed: %s', exc, exc_info=True)
        else:
            print(f'\n  {_C.RED}{SYM_CROSS} Error: {exc}{_C.RESET}')
            print(f'  {_C.DIM}Run with --verbose for full traceback.{_C.RESET}')
        sys.exit(1)
if __name__ == '__main__':
    main()
