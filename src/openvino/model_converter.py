import logging
import subprocess
from pathlib import Path
from typing import Optional
from src.defaults import EMBEDDING_MODEL_ID, OV_EMBEDDING_IR_SUBDIR
logger = logging.getLogger(__name__)
_ONNX_SUBDIR = OV_EMBEDDING_IR_SUBDIR.replace('models/ov/', 'models/onnx/')

def convert_onnx_to_ir(onnx_path: str, output_dir: str, model_name: Optional[str]=None, compress_to_fp16: bool=False) -> Path:
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f'ONNX model not found: {onnx_file}')
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_name = (model_name or onnx_file.stem) + '.xml'
    xml_path = out_dir / xml_name
    try:
        import openvino as ov
        logger.info('Converting %s -> OpenVINO IR (ovc API)', onnx_path)
        try:
            from openvino.tools import ovc
            ov_model = ovc.convert_model(str(onnx_file))
        except (ImportError, AttributeError):
            ov_model = ov.convert_model(str(onnx_file))
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)
        logger.info('IR saved to %s (FP16=%s)', xml_path, compress_to_fp16)
        return xml_path
    except ImportError:
        logger.info('OpenVINO Python API not available, trying mo CLI')
    cmd = ['mo', '--input_model', str(onnx_file), '--output_dir', str(out_dir)]
    if model_name:
        cmd += ['--model_name', model_name]
    if compress_to_fp16:
        cmd += ['--compress_to_fp16']
    logger.info('Running: %s', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error('mo failed:\n%s', result.stderr)
        raise RuntimeError(f'Model Optimizer failed: {result.stderr[:500]}')
    logger.info('IR saved to %s', xml_path)
    return xml_path

def export_sentence_transformer_to_onnx(model_name: str=EMBEDDING_MODEL_ID, output_dir: str=_ONNX_SUBDIR) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = ['optimum-cli', 'export', 'onnx', '--model', model_name, str(out)]
    logger.info('Exporting to ONNX: %s', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error('ONNX export failed:\n%s', result.stderr)
        raise RuntimeError(f'ONNX export failed: {result.stderr[:500]}')
    logger.info('ONNX model exported to %s', out)
    return out

def convert_embedding_model_pipeline(model_name: str=EMBEDDING_MODEL_ID, onnx_dir: str=_ONNX_SUBDIR, ir_dir: str=OV_EMBEDDING_IR_SUBDIR, compress_to_fp16: bool=False) -> Path:
    onnx_out = Path(onnx_dir)
    onnx_model = onnx_out / 'model.onnx'
    if not onnx_model.exists():
        logger.info('Step 1: Exporting %s to ONNX...', model_name)
        export_sentence_transformer_to_onnx(model_name, onnx_dir)
    else:
        logger.info('Step 1: ONNX model already exists at %s, skipping export', onnx_model)
    if not onnx_model.exists():
        raise FileNotFoundError(f'ONNX export completed but model.onnx not found at {onnx_model}. Check the export output directory.')
    logger.info('Step 2: Converting ONNX to OpenVINO IR...')
    xml_path = convert_onnx_to_ir(onnx_path=str(onnx_model), output_dir=ir_dir, compress_to_fp16=compress_to_fp16)
    logger.info('Pipeline complete! IR model at: %s', xml_path)
    return xml_path
