import logging
from pathlib import Path
from typing import Optional, Dict
logger = logging.getLogger(__name__)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class OCREngine:
    SUPPORTED_BACKENDS = {'tesseract', 'paddleocr'}

    def __init__(self, backend: str='tesseract', lang: str='eng'):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown OCR backend '{backend}'. Supported: {self.SUPPORTED_BACKENDS}")
        self.backend = backend
        self.lang = lang
        self._validate_backend()

    def _validate_backend(self) -> None:
        if self.backend == 'tesseract':
            if not PIL_AVAILABLE:
                raise ImportError('Pillow is required for OCR. Install: pip install Pillow')
            if not TESSERACT_AVAILABLE:
                raise ImportError('pytesseract is required. Install: pip install pytesseract\nAlso install Tesseract binary: sudo apt install tesseract-ocr')
        elif self.backend == 'paddleocr':
            logger.warning('PaddleOCR backend is a placeholder. Implement in Phase 2 after Tesseract baseline works.')

    def extract_text(self, image_path: str) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error('Image not found: %s', image_path)
            return ''
        if self.backend == 'tesseract':
            return self._tesseract_extract(image_path)
        elif self.backend == 'paddleocr':
            return self._paddleocr_extract(image_path)
        return ''

    def extract_with_metadata(self, image_path: str) -> Dict:
        text = self.extract_text(image_path)
        return {'text': text, 'backend': self.backend, 'language': self.lang, 'source': str(image_path), 'char_count': len(text)}

    def _tesseract_extract(self, image_path: Path) -> str:
        try:
            img = Image.open(image_path)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            text = pytesseract.image_to_string(img, lang=self.lang)
            logger.info('Tesseract extracted %d chars from %s', len(text), image_path)
            return text.strip()
        except Exception as exc:
            logger.error('Tesseract failed on %s: %s', image_path, exc)
            return ''

    def _paddleocr_extract(self, image_path: Path) -> str:
        try:
            from src.ocr.paddle_engine import PaddleOCREngine
            engine = PaddleOCREngine(lang=self.lang)
            return engine.extract_text(str(image_path))
        except ImportError:
            logger.warning('PaddleOCR is not installed. Returning empty string for %s. Install with: pip install paddleocr paddlepaddle', image_path)
            return ''
        except Exception as exc:
            logger.error('PaddleOCR failed on %s: %s', image_path, exc)
            return ''

def preprocess_image(image_path: str) -> Optional[str]:
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error('OpenCV could not read: %s', image_path)
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        out_path = str(Path(image_path).with_suffix('.preprocessed.png'))
        cv2.imwrite(out_path, thresh)
        logger.info('Preprocessed image saved to %s', out_path)
        return out_path
    except ImportError:
        logger.warning('OpenCV not installed. Skipping preprocessing.')
        return None
    except Exception as exc:
        logger.error('Preprocessing failed for %s: %s', image_path, exc)
        return None
