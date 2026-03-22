import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
from tqdm import tqdm
logger = logging.getLogger(__name__)

@dataclass
class SampledFrame:
    frame_path: str
    timestamp: float
    frame_index: int

class FrameSampler:

    def __init__(self, output_dir: str, interval_seconds: float=5.0, image_format: str='jpg'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = max(0.5, interval_seconds)
        self.image_format = image_format.lower().lstrip('.')

    def sample(self, video_path: str, video_id: Optional[str]=None) -> List[SampledFrame]:
        vp = Path(video_path)
        if not vp.exists():
            logger.error('Video file not found: %s', video_path)
            return []
        if video_id is None:
            video_id = vp.stem
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            logger.error('OpenCV could not open video: %s', video_path)
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        if duration <= 0:
            logger.warning('Video has zero duration: %s', video_path)
            cap.release()
            return []
        frame_interval = int(fps * self.interval_seconds)
        if frame_interval < 1:
            frame_interval = 1
        expected_samples = int(duration / self.interval_seconds) + 1
        video_frame_dir = self.output_dir / video_id
        video_frame_dir.mkdir(parents=True, exist_ok=True)
        frames: List[SampledFrame] = []
        frame_count = 0
        sample_idx = 0
        logger.info('Sampling frames from %s (%.1fs, %.0f fps, interval=%.1fs)', vp.name, duration, fps, self.interval_seconds)
        pbar = tqdm(total=expected_samples, desc=f'Extracting frames: {vp.name}', unit='frame', leave=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                fname = f'{video_id}_frame_{sample_idx:05d}.{self.image_format}'
                fpath = video_frame_dir / fname
                if not fpath.exists():
                    cv2.imwrite(str(fpath), frame)
                frames.append(SampledFrame(frame_path=str(fpath), timestamp=timestamp, frame_index=sample_idx))
                sample_idx += 1
                pbar.update(1)
            frame_count += 1
        pbar.close()
        cap.release()
        logger.info('Extracted %d frames from %s', len(frames), vp.name)
        return frames

    def get_video_duration(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total / fps if fps > 0 else 0.0
