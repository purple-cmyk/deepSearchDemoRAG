import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
logger = logging.getLogger(__name__)

@dataclass
class FrameCaption:
    frame_path: str
    timestamp: float
    caption: str

class FrameCaptioner:

    def __init__(self, model_name: str='Salesforce/blip-image-captioning-base', use_openvino: bool=False, device: str='cpu', max_new_tokens: int=50):
        self.model_name = model_name
        self.use_openvino = use_openvino
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def caption_batch(self, frames, interval: int=1) -> List[FrameCaption]:
        if not frames:
            return []
        model, processor = self._get_model()
        if model is None or processor is None:
            return []
        interval = max(1, interval)
        selected = [f for i, f in enumerate(frames) if i % interval == 0]
        captions: List[FrameCaption] = []
        pbar = tqdm(selected, desc='Captioning frames (BLIP)', unit='frame', leave=False)
        for frame in pbar:
            caption = self._generate_caption(frame.frame_path, model, processor)
            if caption:
                captions.append(FrameCaption(frame_path=frame.frame_path, timestamp=frame.timestamp, caption=caption))
        logger.info('Captioned %d/%d frames (interval=%d)', len(captions), len(frames), interval)
        return captions

    def generate_caption(self, image_path: str) -> str:
        model, processor = self._get_model()
        if model is None or processor is None:
            return ''
        return self._generate_caption(image_path, model, processor)

    def _generate_caption(self, image_path: str, model, processor) -> str:
        try:
            from PIL import Image
        except ImportError:
            logger.error('Pillow is not installed.  pip install Pillow')
            return ''
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as exc:
            logger.debug('Could not open image %s: %s', image_path, exc)
            return ''
        try:
            inputs = processor(images=image, return_tensors='pt')
            if not self.use_openvino:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            caption = processor.decode(outputs[0], skip_special_tokens=True).strip()
            return caption
        except Exception as exc:
            logger.debug('Caption generation failed for %s: %s', image_path, exc)
            return ''

    def _get_model(self):
        if self._model is not None:
            return (self._model, self._processor)
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
        except ImportError:
            logger.error('transformers is not installed.  Install with: pip install transformers')
            return (None, None)
        logger.info('Loading BLIP captioning model: %s (this may take a moment on first run)', self.model_name)
        if self.use_openvino:
            try:
                from optimum.intel import OVModelForVision2Seq
                self._processor = BlipProcessor.from_pretrained(self.model_name)
                self._model = OVModelForVision2Seq.from_pretrained(self.model_name, export=True)
                logger.info('BLIP loaded with OpenVINO acceleration')
                return (self._model, self._processor)
            except ImportError:
                logger.warning('optimum-intel not available — falling back to PyTorch')
            except Exception as exc:
                logger.warning('OpenVINO BLIP load failed (%s) — falling back to PyTorch', exc)
        try:
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            logger.info('BLIP loaded on %s (PyTorch)', self.device)
        except Exception as exc:
            logger.error("Failed to load BLIP model '%s': %s", self.model_name, exc)
            return (None, None)
        return (self._model, self._processor)
