import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
logger = logging.getLogger(__name__)

@dataclass
class VideoChunk:
    timestamp_start: float
    timestamp_end: float
    transcript: str = ''
    caption_text: str = ''
    ocr_text: str = ''
    metadata: Dict = field(default_factory=dict)

@dataclass
class VideoDocument:
    doc_id: str
    source: str
    modality: str = 'video'
    video_path: str = ''
    duration: float = 0.0
    chunks: List[VideoChunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts = []
        for chunk in self.chunks:
            if chunk.caption_text:
                parts.append(chunk.caption_text)
            if chunk.transcript:
                parts.append(chunk.transcript)
            if chunk.ocr_text:
                parts.append(f'[OCR: {chunk.ocr_text}]')
        return '\n'.join(parts)

    def to_dict(self) -> Dict:
        return {'doc_id': self.doc_id, 'source': self.source, 'modality': self.modality, 'video_path': self.video_path, 'duration': self.duration, 'chunks': [asdict(c) for c in self.chunks], 'metadata': self.metadata}

    def to_normalized_dict(self) -> Dict:
        return {'doc_id': self.doc_id, 'source': self.source, 'doc_type': 'video', 'text': self.text, 'image_path': '', 'metadata': {**self.metadata, 'modality': 'video', 'video_path': self.video_path, 'duration': self.duration, 'num_chunks': len(self.chunks)}}

def _make_video_doc_id(video_id: str) -> str:
    return 'video_' + hashlib.sha256(video_id.encode()).hexdigest()[:12]

class VideoDocumentBuilder:

    def __init__(self, output_dir: str, chunk_interval: float=30.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_interval = chunk_interval

    def build_from_captions(self, video_id: str, source: str, captions: List[str], video_path: str='', duration: float=0.0, ocr_results=None, extra_metadata: Optional[Dict]=None) -> VideoDocument:
        doc_id = _make_video_doc_id(video_id)
        ocr_results = ocr_results or []
        if duration <= 0:
            duration = self.chunk_interval
        caption_text = ' '.join((c.strip() for c in captions if c.strip()))
        ocr_text = ' '.join((ocr.text for ocr in ocr_results if hasattr(ocr, 'text'))).strip()
        chunks = []
        if caption_text or ocr_text:
            chunks.append(VideoChunk(timestamp_start=0.0, timestamp_end=round(duration, 2), caption_text=caption_text, ocr_text=ocr_text, metadata={'chunk_index': 0, 'num_captions': len(captions), 'ocr_frames': len(ocr_results)}))
        doc = VideoDocument(doc_id=doc_id, source=source, video_path=video_path, duration=round(duration, 2), chunks=chunks, metadata={**(extra_metadata or {}), 'video_id': video_id, 'num_captions': len(captions), 'total_ocr_frames': len(ocr_results)})
        logger.info('Built caption-mode document: %s (%d captions, %.1fs)', doc_id, len(captions), duration)
        return doc

    def build(self, video_id: str, source: str, transcript_segments, ocr_results, video_path: str='', duration: float=0.0, extra_metadata: Optional[Dict]=None) -> VideoDocument:
        doc_id = _make_video_doc_id(video_id)
        if duration <= 0 and transcript_segments:
            duration = transcript_segments[-1].end
        num_windows = max(1, int(duration / self.chunk_interval) + 1)
        chunks: List[VideoChunk] = []
        for i in range(num_windows):
            t_start = i * self.chunk_interval
            t_end = min((i + 1) * self.chunk_interval, duration)
            transcript_parts = []
            for seg in transcript_segments:
                if seg.end > t_start and seg.start < t_end:
                    transcript_parts.append(seg.text)
            ocr_parts = []
            for ocr in ocr_results:
                if t_start <= ocr.timestamp < t_end:
                    ocr_parts.append(ocr.text)
            transcript_text = ' '.join(transcript_parts).strip()
            ocr_text = ' '.join(ocr_parts).strip()
            if transcript_text or ocr_text:
                chunks.append(VideoChunk(timestamp_start=round(t_start, 2), timestamp_end=round(t_end, 2), transcript=transcript_text, ocr_text=ocr_text, metadata={'chunk_index': i, 'transcript_segments': len(transcript_parts), 'ocr_frames': len(ocr_parts)}))
        doc = VideoDocument(doc_id=doc_id, source=source, video_path=video_path, duration=round(duration, 2), chunks=chunks, metadata={**(extra_metadata or {}), 'video_id': video_id, 'total_transcript_segments': len(transcript_segments), 'total_ocr_frames': len(ocr_results), 'chunk_interval_seconds': self.chunk_interval})
        logger.info('Built video document: %s (%d chunks, %.1fs duration)', doc_id, len(chunks), duration)
        return doc

    def save(self, doc: VideoDocument) -> str:
        out_path = self.output_dir / f'{doc.doc_id}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info('Saved video document: %s', out_path)
        return str(out_path)

    def save_batch(self, docs: List[VideoDocument]) -> List[str]:
        paths = []
        jsonl_path = self.output_dir / 'video_documents.jsonl'
        pbar = tqdm(docs, desc='Saving video documents', unit='doc', leave=False)
        with open(jsonl_path, 'w', encoding='utf-8') as jf:
            for doc in pbar:
                path = self.save(doc)
                paths.append(path)
                jf.write(json.dumps(doc.to_normalized_dict(), ensure_ascii=False) + '\n')
        logger.info('Saved %d video documents + JSONL to %s', len(docs), self.output_dir)
        return paths
