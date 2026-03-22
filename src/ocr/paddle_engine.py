import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class PaddleOCREngine:

    def __init__(self, lang: str='en', use_openvino: bool=False, device: str='CPU', use_angle_cls: bool=True, det_model_dir: Optional[str]=None, rec_model_dir: Optional[str]=None, cls_model_dir: Optional[str]=None, confidence_threshold: int=40):
        self.lang = lang
        self.use_openvino = use_openvino
        self.device = device
        self.use_angle_cls = use_angle_cls
        self.confidence_threshold = confidence_threshold
        self._ocr = None
        if not PADDLE_AVAILABLE:
            logger.warning('PaddleOCR is not installed. Install with: pip install paddleocr paddlepaddle')
            return
        ocr_kwargs = {'use_angle_cls': use_angle_cls, 'lang': lang, 'show_log': False}
        if det_model_dir:
            ocr_kwargs['det_model_dir'] = det_model_dir
        if rec_model_dir:
            ocr_kwargs['rec_model_dir'] = rec_model_dir
        if cls_model_dir:
            ocr_kwargs['cls_model_dir'] = cls_model_dir
        if use_openvino:
            try:
                import openvino
                try:
                    from src.openvino.device_manager import DeviceManager
                    dm = DeviceManager()
                    self.device = dm.select(device)
                    logger.info('OpenVINO device validated via DeviceManager: %s', self.device)
                except Exception as dm_exc:
                    logger.debug('DeviceManager unavailable (%s), using device=%s as-is', dm_exc, device)
                ocr_kwargs['use_onnx'] = True
                logger.info('PaddleOCR initialised with OpenVINO-compatible ONNX backend on %s', self.device)
            except ImportError:
                logger.warning('OpenVINO not installed — PaddleOCR will use default PaddlePaddle backend.')
        try:
            self._ocr = PaddleOCR(**ocr_kwargs)
            logger.info('PaddleOCR engine initialised (lang=%s, openvino=%s)', lang, use_openvino)
        except Exception as exc:
            logger.error('Failed to initialise PaddleOCR: %s', exc)
            self._ocr = None

    def _check_available(self) -> None:
        if self._ocr is None:
            raise RuntimeError('PaddleOCR is not available. Install with: pip install paddleocr paddlepaddle')

    def extract_text(self, image_path: str) -> str:
        self._check_available()
        if not os.path.exists(image_path):
            logger.error('Image not found: %s', image_path)
            return ''
        try:
            results = self._ocr.ocr(image_path, cls=self.use_angle_cls)
            if not results or not results[0]:
                logger.debug('PaddleOCR returned no results for %s', image_path)
                return ''
            lines = []
            for line in results[0]:
                if line is None:
                    continue
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1] * 100
                if confidence >= self.confidence_threshold and text.strip():
                    lines.append(text.strip())
            full_text = ' '.join(lines)
            logger.info('PaddleOCR extracted %d lines (%d chars) from %s', len(lines), len(full_text), image_path)
            return full_text
        except Exception as exc:
            logger.error('PaddleOCR failed for %s: %s', image_path, exc)
            return ''

    def extract_with_boxes(self, image_path: str) -> List[Dict]:
        self._check_available()
        if not os.path.exists(image_path):
            logger.error('Image not found: %s', image_path)
            return []
        try:
            results = self._ocr.ocr(image_path, cls=self.use_angle_cls)
            if not results or not results[0]:
                return []
            boxes = []
            for line in results[0]:
                if line is None:
                    continue
                bbox_points = line[0]
                text_info = line[1]
                text = text_info[0]
                confidence = int(text_info[1] * 100)
                if confidence < self.confidence_threshold or not text.strip():
                    continue
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                left = int(min(xs))
                top = int(min(ys))
                width = int(max(xs) - left)
                height = int(max(ys) - top)
                boxes.append({'text': text.strip(), 'conf': confidence, 'left': left, 'top': top, 'width': width, 'height': height})
            logger.info('PaddleOCR boxes for %s: %d regions (threshold=%d)', image_path, len(boxes), self.confidence_threshold)
            return boxes
        except Exception as exc:
            logger.error('PaddleOCR (boxes) failed for %s: %s', image_path, exc)
            return []

    def batch_extract(self, image_paths: List[str]) -> List[Tuple[str, str]]:
        try:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc='PaddleOCR', unit='img')
        except ImportError:
            iterator = image_paths
        results: List[Tuple[str, str]] = []
        for path in iterator:
            text = self.extract_text(path)
            results.append((path, text))
        logger.info('PaddleOCR batch: processed %d images', len(results))
        return results

    def compare_with_ground_truth(self, image_path: str, ground_truth_words: List[str]) -> Dict[str, float]:
        ocr_text = self.extract_text(image_path)
        ocr_words = set(ocr_text.lower().split())
        gt_words = set((w.lower() for w in ground_truth_words if w.strip()))
        if not gt_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ocr_word_count': len(ocr_words)}
        if not ocr_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ocr_word_count': 0}
        true_positive_p = len(ocr_words & gt_words)
        precision = true_positive_p / len(ocr_words) if ocr_words else 0.0
        true_positive_r = len(ocr_words & gt_words)
        recall = true_positive_r / len(gt_words) if gt_words else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'ocr_word_count': len(ocr_words)}

    def benchmark(self, image_paths: List[str], n_runs: int=3) -> Dict[str, float]:
        import time
        self._check_available()
        times = []
        for run in range(n_runs):
            start = time.perf_counter()
            for path in image_paths:
                self.extract_text(path)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            logger.info('Benchmark run %d: %.2fs for %d images', run + 1, elapsed, len(image_paths))
        import numpy as np_bench
        times_arr = np_bench.array(times)
        return {'engine': 'PaddleOCR', 'openvino': self.use_openvino, 'device': self.device, 'images': len(image_paths), 'mean_time_s': float(times_arr.mean()), 'min_time_s': float(times_arr.min()), 'max_time_s': float(times_arr.max()), 'images_per_sec': float(len(image_paths) / times_arr.mean())}
