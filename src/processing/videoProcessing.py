import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv'}
_TRANSCRIPT_EXTENSIONS = {'.vtt', '.srt', '.txt', '.json'}

@dataclass
class VideoFile:
    video_path: str
    video_id: str
    captions: List[str] = field(default_factory=list)
    transcript_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

class VideoLoader:

    def __init__(self, root: str, max_files: int=0):
        self.root = Path(root)
        self.max_files = max_files

    def discover(self) -> List[VideoFile]:
        if not self.root.exists():
            logger.warning('Video root directory does not exist: %s', self.root)
            return []
        msrvtt_json = self._find_msrvtt_annotation()
        if msrvtt_json is not None:
            logger.info('Detected MSR-VTT dataset layout')
            return self._discover_msrvtt(msrvtt_json)
        logger.info('Using generic video discovery (no MSR-VTT annotation found)')
        return self._discover_generic()

    def _find_msrvtt_annotation(self) -> Optional[Path]:
        candidates = [self.root / 'annotation' / 'MSR_VTT.json', self.root / 'MSR_VTT.json', self.root / 'annotations' / 'MSR_VTT.json']
        for c in candidates:
            if c.exists():
                return c
        return None

    def _discover_msrvtt(self, annotation_path: Path) -> List[VideoFile]:
        logger.info('Loading MSR-VTT annotations from: %s', annotation_path)
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        captions_by_video: Dict[str, List[str]] = {}
        for ann in data.get('annotations', []):
            vid_id = ann.get('image_id', '')
            caption = ann.get('caption', '').strip()
            if vid_id and caption:
                captions_by_video.setdefault(vid_id, []).append(caption)
        logger.info('Loaded %d captions for %d videos', sum((len(v) for v in captions_by_video.values())), len(captions_by_video))
        video_dir = self._find_video_dir()
        if video_dir is None:
            logger.error("Cannot find video directory under %s. Expected 'videos/all/' or similar.", self.root)
            return []
        videos: List[VideoFile] = []
        video_ids = sorted(data.get('images', []), key=lambda x: x.get('id', ''))
        for entry in video_ids:
            vid_id = entry.get('id', '')
            if not vid_id:
                continue
            mp4_path = video_dir / f'{vid_id}.mp4'
            if not mp4_path.exists():
                logger.debug('Video file not found, skipping: %s', mp4_path)
                continue
            if vid_id.startswith('._'):
                continue
            videos.append(VideoFile(video_path=str(mp4_path), video_id=vid_id, captions=captions_by_video.get(vid_id, []), metadata={'dataset': 'msrvtt', 'num_captions': len(captions_by_video.get(vid_id, []))}))
            if self.max_files and len(videos) >= self.max_files:
                logger.info('Reached max_files limit (%d); stopping discovery.', self.max_files)
                break
        logger.info('Discovered %d MSR-VTT video file(s) with captions', len(videos))
        return videos

    def _find_video_dir(self) -> Optional[Path]:
        candidates = [self.root / 'videos' / 'all', self.root / 'videos', self.root / 'video', self.root]
        for c in candidates:
            if c.is_dir():
                for item in c.iterdir():
                    if item.suffix.lower() == '.mp4' and (not item.name.startswith('._')):
                        return c
        return None

    def _discover_generic(self) -> List[VideoFile]:
        videos: List[VideoFile] = []
        for dirpath, _dirnames, filenames in os.walk(self.root):
            dirpath = Path(dirpath)
            for fname in sorted(filenames):
                fpath = dirpath / fname
                if fname.startswith('._'):
                    continue
                if fpath.suffix.lower() not in _VIDEO_EXTENSIONS:
                    continue
                video_id = fpath.stem
                transcript_path = self._find_sidecar(fpath, _TRANSCRIPT_EXTENSIONS)
                metadata_path = self._find_sidecar(fpath, {'.json', '.yaml', '.yml'})
                metadata: Dict = {}
                if metadata_path:
                    metadata = self._try_load_metadata(metadata_path)
                videos.append(VideoFile(video_path=str(fpath), video_id=video_id, transcript_path=transcript_path, metadata_path=metadata_path, metadata=metadata))
                if self.max_files and len(videos) >= self.max_files:
                    logger.info('Reached max_files limit (%d); stopping discovery.', self.max_files)
                    return videos
        logger.info('Discovered %d video file(s) under %s', len(videos), self.root)
        return videos

    @staticmethod
    def _find_sidecar(video_path: Path, extensions: set) -> Optional[str]:
        for ext in extensions:
            candidate = video_path.with_suffix(ext)
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _try_load_metadata(path: str) -> Dict:
        p = Path(path)
        try:
            if p.suffix == '.json':
                return json.loads(p.read_text(encoding='utf-8', errors='replace'))
            elif p.suffix in ('.yaml', '.yml'):
                import yaml
                return yaml.safe_load(p.read_text(encoding='utf-8', errors='replace')) or {}
        except Exception as exc:
            logger.debug('Could not parse metadata %s: %s', path, exc)
        return {}
