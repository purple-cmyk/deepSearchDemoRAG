import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from src.defaults import CLIP_MODEL_ID
logger = logging.getLogger(__name__)
CLIP_DIMENSION = 512
DEFAULT_MODEL_NAME = CLIP_MODEL_ID
try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class CLIPEncoder:

    def __init__(self, model_name: str=DEFAULT_MODEL_NAME, use_openvino: bool=False, device: str='CPU'):
        self.model_name = model_name
        self.use_openvino = use_openvino
        self.device = device
        self._dim = CLIP_DIMENSION
        self._model = None
        self._processor = None
        self._ov_vision_model = None
        if not CLIP_AVAILABLE:
            logger.warning('transformers CLIP classes not available. Install: pip install transformers')
            return
        if not PIL_AVAILABLE:
            logger.warning('Pillow is required for image processing. Install: pip install Pillow')
            return
        try:
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(model_name)
            self._model.eval()
            logger.info('CLIP model loaded: %s', model_name)
        except Exception as exc:
            logger.error("Failed to load CLIP model '%s': %s", model_name, exc)
            self._model = None
            self._processor = None
            return
        if use_openvino:
            self._try_openvino_vision(device)

    def _try_openvino_vision(self, device: str) -> None:
        try:
            import torch
            import openvino as ov
            try:
                from src.openvino.device_manager import DeviceManager
                dm = DeviceManager()
                device = dm.select(device)
            except Exception:
                pass
            self.device = device
            dummy_image = Image.new('RGB', (224, 224))
            inputs = self._processor(images=dummy_image, return_tensors='pt')
            pixel_values = inputs['pixel_values']
            import io
            buffer = io.BytesIO()
            vision_model = self._model.vision_model

            class VisionWrapper(torch.nn.Module):

                def __init__(self, vision_model, visual_projection):
                    super().__init__()
                    self.vision_model = vision_model
                    self.visual_projection = visual_projection

                def forward(self, pixel_values):
                    vision_outputs = self.vision_model(pixel_values=pixel_values)
                    pooled = vision_outputs.pooler_output
                    return self.visual_projection(pooled)
            wrapper = VisionWrapper(self._model.vision_model, self._model.visual_projection)
            wrapper.eval()
            with torch.no_grad():
                ov_model = ov.convert_model(wrapper, example_input=pixel_values)
            core = ov.Core()
            self._ov_vision_model = core.compile_model(ov_model, device)
            logger.info('CLIP vision model compiled with OpenVINO on %s', device)
        except Exception as exc:
            logger.warning('OpenVINO acceleration for CLIP vision failed (%s). Falling back to PyTorch.', exc)
            self._ov_vision_model = None

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_available(self) -> bool:
        return self._model is not None and self._processor is not None

    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        if not self.is_available:
            logger.warning('CLIP encoder not available')
            return None
        path = Path(image_path)
        if not path.exists():
            logger.error('Image not found: %s', image_path)
            return None
        try:
            image = Image.open(path).convert('RGB')
            if self._ov_vision_model is not None:
                inputs = self._processor(images=image, return_tensors='np')
                pixel_values = inputs['pixel_values'].astype(np.float32)
                result = self._ov_vision_model({'pixel_values': pixel_values})
                embedding = result[0].squeeze()
            else:
                import torch
                inputs = self._processor(images=image, return_tensors='pt')
                with torch.no_grad():
                    image_features = self._model.get_image_features(**inputs)
                embedding = image_features.squeeze().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 1e-09:
                embedding = embedding / norm
            return embedding.astype(np.float32)
        except Exception as exc:
            logger.error('CLIP image encoding failed for %s: %s', image_path, exc)
            return None

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        embeddings = np.zeros((len(image_paths), self._dim), dtype=np.float32)
        for i, path in enumerate(image_paths):
            emb = self.encode_image(path)
            if emb is not None:
                embeddings[i] = emb
        return embeddings

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        if not self.is_available:
            logger.warning('CLIP encoder not available')
            return None
        try:
            import torch
            inputs = self._processor(text=[text], return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
            embedding = text_features.squeeze().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 1e-09:
                embedding = embedding / norm
            return embedding.astype(np.float32)
        except Exception as exc:
            logger.error('CLIP text encoding failed: %s', exc)
            return None

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            emb = self.encode_text(text)
            if emb is not None:
                embeddings[i] = emb
        return embeddings
