import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from src.ingestion.loader import RawDocument
logger = logging.getLogger(__name__)

def _make_doc_id(key: str) -> str:
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]

def _detect_language(text: str) -> Optional[str]:
    if not text or len(text.strip()) < 20:
        return None
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        logger.debug('langdetect is not installed -- skipping language detection.  Install it with:  pip install langdetect')
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None
_BOILERPLATE_PATTERNS: List[re.Pattern] = [re.compile('(?i)^\\s*page\\s+\\d+(?:\\s+of\\s+\\d+)?\\s*$', re.MULTILINE), re.compile('^\\s*-?\\s*\\d{1,4}\\s*-?\\s*$', re.MULTILINE), re.compile('(?i)^\\s*(?:confidential|privileged|do not distribute|internal use only|proprietary|strictly private|not for public release).*$', re.MULTILINE), re.compile('(?i)^\\s*-{0,3}\\s*draft\\s*-{0,3}\\s*$', re.MULTILINE), re.compile('(?i)^\\s*(?:from:|to:|cc:|bcc:|sent:|date:|subject:|fax:)\\s*$', re.MULTILINE)]

def _strip_boilerplate(text: str) -> str:
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub('', text)
    text = re.sub('\\n{3,}', '\n\n', text)
    return text.strip()

class NormalizedDocument:

    def __init__(self, doc_id: str, source: str, doc_type: str, text: str, image_path: str, metadata: Dict):
        self.doc_id = doc_id
        self.source = source
        self.doc_type = doc_type
        self.text = text
        self.image_path = image_path
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"NormalizedDocument(id='{self.doc_id}', type='{self.doc_type}', text_len={len(self.text)})"

    def to_dict(self) -> Dict:
        return {'doc_id': self.doc_id, 'source': self.source, 'doc_type': self.doc_type, 'text': self.text, 'image_path': self.image_path, 'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data: Dict) -> 'NormalizedDocument':
        return cls(doc_id=data['doc_id'], source=data['source'], doc_type=data['doc_type'], text=data['text'], image_path=data.get('image_path', ''), metadata=data.get('metadata', {}))

class DocumentNormalizer:

    def __init__(self, output_dir: str='data/processed/normalised', deduplicate: bool=True, detect_language: bool=True, strip_boilerplate: bool=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deduplicate = deduplicate
        self.detect_language = detect_language
        self.strip_boilerplate = strip_boilerplate

    def _load_existing_ids(self, filename: str='documents.jsonl') -> Set[str]:
        out_path = self.output_dir / filename
        ids: Set[str] = set()
        if not out_path.exists():
            return ids
        with open(out_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)['doc_id'])
                    except (json.JSONDecodeError, KeyError):
                        continue
        logger.info('Dedup: %d existing document IDs loaded from %s', len(ids), out_path)
        return ids

    def normalize(self, raw_docs: List[RawDocument], jsonl_filename: str='documents.jsonl') -> List[NormalizedDocument]:
        existing_ids: Set[str] = set()
        if self.deduplicate:
            existing_ids = self._load_existing_ids(jsonl_filename)
        try:
            from tqdm import tqdm
            iterator = tqdm(raw_docs, desc='Normalizing', unit='doc')
        except ImportError:
            iterator = raw_docs
        normalized: List[NormalizedDocument] = []
        skipped = 0
        for raw in iterator:
            doc_id = _make_doc_id(raw.doc_key)
            if doc_id in existing_ids:
                skipped += 1
                continue
            existing_ids.add(doc_id)
            text = raw.text.strip()
            if text and self.strip_boilerplate:
                text = _strip_boilerplate(text)
            metadata = dict(raw.metadata)
            if self.detect_language:
                lang = _detect_language(text)
                metadata['language'] = lang or 'en'
            doc_type = self._classify_type(raw)
            norm = NormalizedDocument(doc_id=doc_id, source=raw.source, doc_type=doc_type, text=text, image_path=raw.image_path or '', metadata=metadata)
            normalized.append(norm)
        if skipped:
            logger.info('Dedup: skipped %d duplicate documents', skipped)
        logger.info('Normalised %d documents', len(normalized))
        return normalized

    def save(self, docs: List[NormalizedDocument], filename: str='documents.jsonl') -> Path:
        out_path = self.output_dir / filename
        with open(out_path, 'w', encoding='utf-8') as fh:
            for doc in docs:
                fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
        logger.info('Saved %d normalised documents to %s', len(docs), out_path)
        return out_path

    def load(self, filename: str='documents.jsonl') -> List[NormalizedDocument]:
        out_path = self.output_dir / filename
        if not out_path.exists():
            logger.warning('No normalised documents found at %s', out_path)
            return []
        docs: List[NormalizedDocument] = []
        with open(out_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(NormalizedDocument.from_dict(json.loads(line)))
        logger.info('Loaded %d normalised documents from %s', len(docs), out_path)
        return docs

    @staticmethod
    def _classify_type(raw: RawDocument) -> str:
        dataset = raw.metadata.get('dataset', '')
        if dataset == 'funsd':
            return 'form'
        elif dataset == 'docvqa':
            return 'document_image'
        elif dataset == 'rvl_cdip':
            return 'classified_image'
        file_type = raw.metadata.get('file_type', '')
        if file_type == '.docx':
            return 'docx'
        elif file_type == '.pptx':
            return 'pptx'
        elif file_type == '.pdf':
            return 'pdf'
        if raw.image_path:
            return 'image'
        if raw.text:
            return 'text'
        return 'unknown'
