import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
logger = logging.getLogger(__name__)

@dataclass
class FrameOCRResult:
    frame_path: str
    timestamp: float
    text: str
    word_count: int

class FrameOCR:

    def __init__(self, lang: str='eng', min_word_count: int=3, preprocess: bool=True, confidence_threshold: int=40):
        self.lang = lang
        self.min_word_count = min_word_count
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold

    def extract_batch(self, frames) -> List[FrameOCRResult]:
        if not frames:
            return []
        try:
            import pytesseract
        except ImportError:
            logger.error('pytesseract is not installed.  Install with: pip install pytesseract')
            return []
        results: List[FrameOCRResult] = []
        pbar = tqdm(frames, desc='Running OCR on frames', unit='frame', leave=False)
        for frame in pbar:
            text = self._extract_text(frame.frame_path, pytesseract)
            words = text.split()
            word_count = len(words)
            if word_count >= self.min_word_count:
                results.append(FrameOCRResult(frame_path=frame.frame_path, timestamp=frame.timestamp, text=text, word_count=word_count))
        logger.info('OCR completed: %d/%d frames had meaningful text (>=%d words)', len(results), len(frames), self.min_word_count)
        return results

    def _extract_text(self, image_path: str, pytesseract_module) -> str:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            logger.debug('Could not read image: %s', image_path)
            return ''
        if self.preprocess:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        try:
            data = pytesseract_module.image_to_data(img, lang=self.lang, output_type=pytesseract_module.Output.DICT)
            words = []
            for i, conf in enumerate(data['conf']):
                try:
                    if int(conf) >= self.confidence_threshold:
                        word = data['text'][i].strip()
                        if word:
                            words.append(word)
                except (ValueError, TypeError):
                    continue
            return ' '.join(words)
        except Exception as exc:
            logger.debug('OCR failed on %s: %s', image_path, exc)
            return ''
