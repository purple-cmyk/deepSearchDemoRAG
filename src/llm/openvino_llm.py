import logging
import time
from pathlib import Path
from typing import Dict, Optional
logger = logging.getLogger(__name__)
RAG_SYSTEM_PROMPT = 'You are a helpful AI assistant that answers questions based on the provided context.\nUse ONLY the information from the context below to answer the question.\nIf the context does not contain enough information to answer, say so honestly.\nDo not make up information.\n\nCONTEXT:\n{context}'
RAG_USER_TEMPLATE = 'Based on the context provided, please answer the following question:\n\n{question}'
RAG_PROMPT_LEAN = 'Answer using context only:\n{context}\nQ: {question}\nA:'

class OVLLMClient:

    def __init__(self, model_dir: str='', device: str='CPU'):
        self.model_dir = model_dir
        self.device = device
        self._pipeline = None
        self._ov_model = None
        self._tokenizer = None
        self._backend = None
        if model_dir:
            self._try_load(model_dir, device)
        else:
            logger.info('OVLLMClient created without model_dir — call is_available() before generate().')

    def _try_load(self, model_dir: str, device: str) -> None:
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.warning('Model directory does not exist: %s', model_dir)
            return
        try:
            import openvino_genai
            self._pipeline = openvino_genai.LLMPipeline(str(model_path), device)
            self._backend = 'genai'
            logger.info('Loaded OpenVINO LLM (genai backend) from %s on %s', model_dir, device)
            return
        except ImportError:
            logger.debug('openvino-genai not installed, trying optimum fallback')
        except Exception as exc:
            logger.warning('openvino_genai failed: %s — trying optimum', exc)
        try:
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer
            self._ov_model = OVModelForCausalLM.from_pretrained(str(model_path), device=device)
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._backend = 'optimum'
            logger.info('Loaded OpenVINO LLM (optimum backend) from %s on %s', model_dir, device)
            return
        except ImportError:
            logger.warning('Neither openvino-genai nor optimum-intel installed. Install one of: pip install openvino-genai | pip install optimum-intel[openvino]')
        except Exception as exc:
            logger.warning('optimum.intel failed: %s', exc)
        logger.warning('OVLLMClient could not load model from %s — generate() will return a fallback message.', model_dir)

    def is_available(self) -> bool:
        return self._backend is not None

    @staticmethod
    def build_rag_prompt(question: str, context: str, template: str='lean') -> Dict[str, str]:
        if template == 'lean':
            if context:
                user_msg = RAG_PROMPT_LEAN.format(context=context, question=question)
            else:
                user_msg = f'Q: {question}\nA:'
            return {'system': '', 'user': user_msg}
        system_msg = RAG_SYSTEM_PROMPT.format(context=context) if context else ''
        user_msg = RAG_USER_TEMPLATE.format(question=question)
        return {'system': system_msg, 'user': user_msg}

    def generate(self, question: str, context: str='', temperature: float=0.3, top_p: float=0.9, max_tokens: int=1024, template: str='default', preset: Optional[str]=None, debug: bool=False) -> str:
        prompt_parts = self.build_rag_prompt(question, context, template)
        system_msg = prompt_parts['system']
        user_msg = prompt_parts['user']
        if debug:
            print('\n' + '=' * 60)
            print('DEBUG: OpenVINO LLM Prompt')
            print('=' * 60)
            print(f'Backend: {self._backend}')
            print(f'Temperature: {temperature}, Top-p: {top_p}')
            print(f'--- SYSTEM ---\n{system_msg}')
            print(f'--- USER ---\n{user_msg}')
            print('=' * 60 + '\n')
        if system_msg:
            full_prompt = f'[INST] {system_msg}\n\n{user_msg} [/INST]'
        else:
            full_prompt = f'[INST] {user_msg} [/INST]'
        if self._backend == 'genai':
            return self._generate_genai(full_prompt, temperature, top_p, max_tokens)
        if self._backend == 'optimum':
            return self._generate_optimum(full_prompt, temperature, top_p, max_tokens)
        return "[OVLLMClient] Model not loaded. To use OpenVINO for LLM inference:\n  1. pip install openvino-genai\n  2. Convert model: optimum-cli export openvino --model mistralai/Mistral-7B-Instruct-v0.2 --weight-format int4 models/ov/mistral-7b-instruct/\n  3. Pass model_dir='models/ov/mistral-7b-instruct' to OVLLMClient\n\nFalling back: use 'python cli.py ask' with Ollama instead."

    def _generate_genai(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        try:
            import openvino_genai
            config = openvino_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
            config.temperature = temperature
            config.top_p = top_p
            config.do_sample = temperature > 0
            start = time.perf_counter()
            result = self._pipeline.generate(prompt, config)
            elapsed = time.perf_counter() - start
            answer = str(result).strip()
            logger.info("OpenVINO LLM (genai): %d chars in %.1fs for: '%s'", len(answer), elapsed, prompt[:60])
            return answer
        except Exception as exc:
            logger.error('OpenVINO genai generation failed: %s', exc)
            return f'[ERROR] OpenVINO generation failed: {exc}'

    def _generate_optimum(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        try:
            start = time.perf_counter()
            inputs = self._tokenizer(prompt, return_tensors='pt')
            input_len = inputs['input_ids'].shape[1]
            gen_kwargs = {'max_new_tokens': max_tokens, 'do_sample': temperature > 0}
            if temperature > 0:
                gen_kwargs['temperature'] = temperature
                gen_kwargs['top_p'] = top_p
            output_ids = self._ov_model.generate(**inputs, **gen_kwargs)
            new_tokens = output_ids[0][input_len:]
            answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            elapsed = time.perf_counter() - start
            logger.info("OpenVINO LLM (optimum): %d chars in %.1fs for: '%s'", len(answer), elapsed, prompt[:60])
            return answer
        except Exception as exc:
            logger.error('OpenVINO optimum generation failed: %s', exc)
            return f'[ERROR] OpenVINO generation failed: {exc}'

    def generate_stream(self, question: str, context: str='', temperature: float=0.3, top_p: float=0.9, max_tokens: int=1024, template: str='lean', preset: Optional[str]=None):
        prompt_parts = self.build_rag_prompt(question, context, template)
        system_msg = prompt_parts['system']
        user_msg = prompt_parts['user']
        if system_msg:
            full_prompt = f'[INST] {system_msg}\n\n{user_msg} [/INST]'
        else:
            full_prompt = f'[INST] {user_msg} [/INST]'
        if not self.is_available():
            yield '[OVLLMClient] Model not loaded.  Run: optimum-cli export openvino --model mistralai/Mistral-7B-Instruct-v0.2 --weight-format int4 models/ov/mistral-7b-instruct/'
            return
        if self._backend == 'genai':
            try:
                import openvino_genai
                config = openvino_genai.GenerationConfig()
                config.max_new_tokens = max_tokens
                config.temperature = temperature
                config.top_p = top_p
                config.do_sample = temperature > 0
                tokens_collected: list = []

                class _Streamer(openvino_genai.StreamerBase):

                    def put(self, token_id: int) -> bool:
                        word = self._pipe.get_tokenizer().decode([token_id])
                        tokens_collected.append(word)
                        return False
                try:
                    streamer = _Streamer()
                    if hasattr(streamer, '_pipe') is False:
                        streamer._pipe = self._pipeline
                    self._pipeline.generate(full_prompt, config, streamer)
                    for tok in tokens_collected:
                        yield tok
                    return
                except (AttributeError, TypeError):
                    result = self._pipeline.generate(full_prompt, config)
                    yield str(result).strip()
                    return
            except Exception as exc:
                yield f'\n[ERROR] OpenVINO genai streaming failed: {exc}'
                return
        if self._backend == 'optimum':
            try:
                result = self._generate_optimum(full_prompt, temperature, top_p, max_tokens)
                yield result
            except Exception as exc:
                yield f'\n[ERROR] OpenVINO optimum streaming failed: {exc}'

    def benchmark(self, prompt: str='What is machine learning?', max_tokens: int=100, n_runs: int=3) -> Dict[str, float]:
        if not self.is_available():
            return {'error': 'Model not loaded'}
        times = []
        char_counts = []
        for i in range(n_runs):
            start = time.perf_counter()
            result = self.generate(question=prompt, max_tokens=max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            char_counts.append(len(result))
            logger.info('Benchmark run %d: %.2fs, %d chars', i + 1, elapsed, len(result))
        import numpy as np
        times_arr = np.array(times)
        chars_arr = np.array(char_counts)
        return {'backend': self._backend, 'device': self.device, 'mean_time_s': float(times_arr.mean()), 'min_time_s': float(times_arr.min()), 'max_time_s': float(times_arr.max()), 'mean_chars': float(chars_arr.mean()), 'chars_per_sec': float(chars_arr.mean() / times_arr.mean())}
