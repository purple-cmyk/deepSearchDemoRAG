from src.ocr.tesseract_engine import TesseractEngine
try:
    from src.ocr.paddle_engine import PaddleOCREngine
except ImportError:
    pass
