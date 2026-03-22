import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class TesseractEngine:

    def __init__(self, lang: str='eng', preprocess: bool=True, confidence_threshold: int=40, psm: int=3):
        if not TESSERACT_AVAILABLE:
            raise ImportError('pytesseract is required.  Install: pip install pytesseract  Also install the system binary: sudo apt install tesseract-ocr')
        if not PIL_AVAILABLE:
            raise ImportError('Pillow is required.  Install: pip install pillow')
        self.lang = lang
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold
        self.psm = psm

    def extract_text(self, image_path: str) -> str:
        img = self._load_image(image_path)
        if img is None:
            return ''
        if self.preprocess and CV2_AVAILABLE:
            img = self._preprocess(img)
        try:
            config = f'--psm {self.psm}'
            text = pytesseract.image_to_string(img, lang=self.lang, config=config)
            return text.strip()
        except Exception as exc:
            logger.error('OCR failed for %s: %s', image_path, exc)
            return ''

    def extract_with_boxes(self, image_path: str) -> List[Dict]:
        img = self._load_image(image_path)
        if img is None:
            return []
        if self.preprocess and CV2_AVAILABLE:
            img = self._preprocess(img)
        try:
            config = f'--psm {self.psm}'
            data = pytesseract.image_to_data(img, lang=self.lang, config=config, output_type=pytesseract.Output.DICT)
        except Exception as exc:
            logger.error('OCR (boxes) failed for %s: %s', image_path, exc)
            return []
        results: List[Dict] = []
        n_words = len(data['text'])
        for i in range(n_words):
            conf = int(data['conf'][i])
            word = data['text'][i].strip()
            if conf < self.confidence_threshold or not word:
                continue
            results.append({'text': word, 'conf': conf, 'left': data['left'][i], 'top': data['top'][i], 'width': data['width'][i], 'height': data['height'][i]})
        logger.info('OCR boxes for %s: %d words (threshold=%d)', image_path, len(results), self.confidence_threshold)
        return results

    def batch_extract(self, image_paths: List[str]) -> List[Tuple[str, str]]:
        try:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc='OCR', unit='img')
        except ImportError:
            iterator = image_paths
        results: List[Tuple[str, str]] = []
        for path in iterator:
            text = self.extract_text(path)
            results.append((path, text))
        logger.info('Batch OCR: processed %d images', len(results))
        return results

    def compare_with_ground_truth(self, image_path: str, ground_truth_words: List[str]) -> Dict:
        ocr_text = self.extract_text(image_path)
        ocr_words = [w.lower() for w in ocr_text.split() if w.strip()]
        gt_words = [w.lower() for w in ground_truth_words if w.strip()]
        gt_set = set(gt_words)
        ocr_set = set(ocr_words)
        matches = gt_set & ocr_set
        precision = len(matches) / len(ocr_set) if ocr_set else 0.0
        recall = len(matches) / len(gt_set) if gt_set else 0.0
        return {'ocr_text': ocr_text, 'ocr_words': ocr_words, 'ground_truth': gt_words, 'word_accuracy': recall * 100, 'precision': precision * 100, 'recall': recall * 100, 'ocr_word_count': len(ocr_words), 'gt_word_count': len(gt_words)}

    def tune_settings(self, image_path: str, ground_truth_words: List[str], test_configs: Optional[List[Dict]]=None) -> List[Dict]:
        if test_configs is None:
            test_configs = [{'preprocess': False, 'confidence_threshold': 40, 'psm': 3}, {'preprocess': True, 'confidence_threshold': 40, 'psm': 3}, {'preprocess': True, 'confidence_threshold': 30, 'psm': 3}, {'preprocess': True, 'confidence_threshold': 50, 'psm': 3}, {'preprocess': True, 'confidence_threshold': 40, 'psm': 6}, {'preprocess': True, 'confidence_threshold': 40, 'psm': 11}]
        results: List[Dict] = []
        for cfg in test_configs:
            engine = TesseractEngine(lang=self.lang, preprocess=cfg.get('preprocess', True), confidence_threshold=cfg.get('confidence_threshold', 40), psm=cfg.get('psm', 3))
            comparison = engine.compare_with_ground_truth(image_path, ground_truth_words)
            results.append({'config': cfg, 'word_accuracy': comparison['word_accuracy'], 'precision': comparison['precision'], 'recall': comparison['recall'], 'ocr_word_count': comparison['ocr_word_count']})
        results.sort(key=lambda r: r['recall'], reverse=True)
        logger.info('Tuned %d configs for %s. Best recall: %.1f%%', len(results), image_path, results[0]['recall'])
        return results

    @staticmethod
    def _load_image(image_path: str) -> Optional['Image.Image']:
        p = Path(image_path)
        if not p.exists():
            logger.warning('Image not found: %s', image_path)
            return None
        try:
            return Image.open(p)
        except Exception as exc:
            logger.error('Failed to open image %s: %s', image_path, exc)
            return None

    @staticmethod
    def _preprocess(img: 'Image.Image') -> 'Image.Image':
        if not CV2_AVAILABLE:
            return img
        gray = np.array(img.convert('L'))
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
        return Image.fromarray(binary)
