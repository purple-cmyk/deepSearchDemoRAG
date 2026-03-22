import logging
import tempfile
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)

class AudioExtractor:

    def __init__(self, output_dir: str, sample_rate: int=16000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

    def extract(self, video_path: str, output_filename: Optional[str]=None) -> Optional[str]:
        vp = Path(video_path)
        if not vp.exists():
            logger.error('Video file not found: %s', video_path)
            return None
        if output_filename is None:
            output_filename = vp.stem + '.wav'
        wav_path = self.output_dir / output_filename
        if wav_path.exists():
            logger.debug('Audio already extracted: %s', wav_path)
            return str(wav_path)
        try:
            from moviepy import VideoFileClip
        except ImportError:
            logger.error('moviepy is not installed.  Install with: pip install moviepy')
            return None
        try:
            logger.debug('Extracting audio: %s -> %s', video_path, wav_path)
            clip = VideoFileClip(str(vp))
            if clip.audio is None:
                logger.warning('Video has no audio track: %s', video_path)
                clip.close()
                return None
            clip.audio.write_audiofile(str(wav_path), fps=self.sample_rate, nbytes=2, codec='pcm_s16le', logger=None)
            clip.close()
            logger.info('Audio extracted: %s (%.1f KB)', wav_path.name, wav_path.stat().st_size / 1024)
            return str(wav_path)
        except Exception as exc:
            logger.error('Audio extraction failed for %s: %s', video_path, exc)
            return None
