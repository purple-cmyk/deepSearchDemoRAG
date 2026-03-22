import json
import logging
import time
import sys
from typing import Optional, Dict, List, Generator
logger = logging.getLogger(__name__)
try:
    import urllib.request
    import urllib.error
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
DEFAULT_OLLAMA_URL = 'http://localhost:11434'
DEFAULT_MODEL = 'mistral'
RAG_SYSTEM_PROMPT = 'You are a helpful AI assistant that answers questions based on the provided context.\nUse ONLY the information from the context below to answer the question.\nIf the context does not contain enough information to answer, say so honestly.\nDo not make up information.\n\nCONTEXT:\n{context}\n'
RAG_USER_TEMPLATE = 'Based on the context provided, please answer the following question:\n\n{question}'
RAG_SYSTEM_PROMPT_CITED = 'You are a precise document analysis assistant.\nAnswer questions using ONLY the numbered sources below.\nFor every claim in your answer, cite the source in [Source N] format.\nIf no source supports an answer, say "The provided documents do not contain this information."\n\nSOURCES:\n{context}\n'
RAG_USER_TEMPLATE_CITED = 'Question: {question}\n\nProvide a clear, sourced answer:'
RAG_SYSTEM_PROMPT_CONCISE = 'Extract the answer from the context below.\nReply with ONLY the answer — no explanation, no preamble.\nIf the answer is not in the context, reply "Not found."\n\nContext:\n{context}\n'
RAG_USER_TEMPLATE_CONCISE = '{question}'
RAG_PROMPT_LEAN = 'Answer using context only:\n{context}\nQ: {question}\nA:'
PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {'default': {'system': RAG_SYSTEM_PROMPT, 'user': RAG_USER_TEMPLATE, 'description': 'Balanced: grounded answers, admits uncertainty'}, 'cited': {'system': RAG_SYSTEM_PROMPT_CITED, 'user': RAG_USER_TEMPLATE_CITED, 'description': 'Strict citations: every claim references [Source N]'}, 'concise': {'system': RAG_SYSTEM_PROMPT_CONCISE, 'user': RAG_USER_TEMPLATE_CONCISE, 'description': 'Short factual answers: dates, names, amounts'}, 'lean': {'system': '', 'user': RAG_PROMPT_LEAN, 'description': 'Lean: minimal tokens, fastest for CLI streaming'}}
GENERATION_PRESETS: Dict[str, Dict] = {'precise': {'temperature': 0.1, 'top_p': 0.5, 'description': 'Most deterministic. Best for factual Q&A, dates, amounts.'}, 'balanced': {'temperature': 0.3, 'top_p': 0.9, 'description': 'Default RAG setting. Grounded but natural language.'}, 'creative': {'temperature': 0.7, 'top_p': 0.95, 'description': 'More varied responses. Good for summaries, explanations.'}, 'exploratory': {'temperature': 1.0, 'top_p': 1.0, 'description': 'Maximum diversity. Not recommended for RAG — may hallucinate.'}}

class OllamaClient:

    def __init__(self, base_url: str=DEFAULT_OLLAMA_URL, model: str=DEFAULT_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def is_available(self) -> bool:
        try:
            url = f'{self.base_url}/api/tags'
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                models = [m['name'] for m in data.get('models', [])]
                available = any((self.model in m for m in models))
                if not available:
                    logger.warning("Model '%s' not found. Available: %s. Run: ollama pull %s", self.model, models, self.model)
                return available
        except Exception as exc:
            logger.error('Ollama not reachable at %s: %s', self.base_url, exc)
            return False

    def list_models(self) -> List[str]:
        try:
            url = f'{self.base_url}/api/tags'
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            return []

    @staticmethod
    def build_rag_prompt(question: str, context: str, template: str='lean') -> Dict[str, str]:
        tmpl = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES['lean'])
        if template == 'lean':
            if context:
                user_msg = tmpl['user'].format(context=context, question=question)
            else:
                user_msg = f'Q: {question}\nA:'
            return {'system': '', 'user': user_msg}
        system_msg = tmpl['system'].format(context=context) if context else ''
        user_msg = tmpl['user'].format(question=question)
        return {'system': system_msg, 'user': user_msg}

    def generate(self, question: str, context: str='', temperature: float=0.3, top_p: float=0.9, max_tokens: int=1024, template: str='default', preset: Optional[str]=None, debug: bool=False) -> str:
        if preset and preset in GENERATION_PRESETS:
            p = GENERATION_PRESETS[preset]
            temperature = p['temperature']
            top_p = p['top_p']
            logger.info("Using preset '%s': temp=%.1f, top_p=%.2f", preset, temperature, top_p)
        prompt_parts = self.build_rag_prompt(question, context, template)
        system_msg = prompt_parts['system']
        user_msg = prompt_parts['user']
        if debug:
            print('\n' + '=' * 60)
            print('DEBUG: RAG Prompt Sent to LLM')
            print('=' * 60)
            print(f'Template: {template}')
            print(f'Temperature: {temperature}, Top-p: {top_p}')
            print(f'--- SYSTEM ---\n{system_msg}')
            print(f'--- USER ---\n{user_msg}')
            print('=' * 60 + '\n')
        payload = {'model': self.model, 'prompt': user_msg, 'system': system_msg, 'stream': False, 'options': {'temperature': temperature, 'top_p': top_p, 'num_predict': max_tokens}}
        try:
            url = f'{self.base_url}/api/generate'
            body = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            elapsed = time.time() - start_time
            answer = result.get('response', '').strip()
            eval_count = result.get('eval_count', 0)
            eval_duration = result.get('eval_duration', 0)
            tokens_per_sec = eval_count / (eval_duration / 1000000000.0) if eval_duration else 0
            logger.info("Generated answer (%d chars, %d tokens, %.1f tok/s, %.1fs) for question: '%s'", len(answer), eval_count, tokens_per_sec, elapsed, question[:60])
            return answer
        except urllib.error.URLError as exc:
            logger.error('Ollama request failed: %s', exc)
            return f'[ERROR] Could not reach Ollama at {self.base_url}. Is it running?'
        except Exception as exc:
            logger.error('LLM generation failed: %s', exc)
            return f'[ERROR] Generation failed: {exc}'

    def generate_stream(self, question: str, context: str='', temperature: float=0.3, top_p: float=0.9, max_tokens: int=1024, template: str='default', preset: Optional[str]=None) -> Generator[str, None, None]:
        if preset and preset in GENERATION_PRESETS:
            p = GENERATION_PRESETS[preset]
            temperature = p['temperature']
            top_p = p['top_p']
        prompt_parts = self.build_rag_prompt(question, context, template)
        payload = {'model': self.model, 'prompt': prompt_parts['user'], 'system': prompt_parts['system'], 'stream': True, 'options': {'temperature': temperature, 'top_p': top_p, 'num_predict': max_tokens}}
        try:
            url = f'{self.base_url}/api/generate'
            body = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        token = chunk.get('response', '')
                        if token:
                            yield token
                        if chunk.get('done', False):
                            eval_count = chunk.get('eval_count', 0)
                            eval_duration = chunk.get('eval_duration', 0)
                            tps = eval_count / (eval_duration / 1000000000.0) if eval_duration else 0
                            logger.info('Stream complete: %d tokens, %.1f tok/s', eval_count, tps)
                            return
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as exc:
            yield f'\n[ERROR] Could not reach Ollama at {self.base_url}: {exc}'
        except Exception as exc:
            yield f'\n[ERROR] Streaming failed: {exc}'

    def generate_without_context(self, prompt: str, temperature: float=0.7, max_tokens: int=512) -> str:
        payload = {'model': self.model, 'prompt': prompt, 'stream': False, 'options': {'temperature': temperature, 'num_predict': max_tokens}}
        try:
            url = f'{self.base_url}/api/generate'
            body = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get('response', '').strip()
        except Exception as exc:
            logger.error('LLM generation (no context) failed: %s', exc)
            return f'[ERROR] Generation failed: {exc}'

    @staticmethod
    def measure_answer_quality(answer: str, context: str, question: str) -> Dict[str, float]:
        import re

        def tokenise(text: str) -> set:
            return set(re.findall('\\w+', text.lower()))
        answer_words = tokenise(answer)
        context_words = tokenise(context)
        question_words = tokenise(question)
        stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        answer_content = answer_words - stop
        context_content = context_words - stop
        question_content = question_words - stop
        if answer_content:
            groundedness = len(answer_content & context_content) / len(answer_content)
        else:
            groundedness = 0.0
        if question_content:
            relevance = len(question_content & answer_content) / len(question_content)
        else:
            relevance = 0.0
        length_ratio = len(answer) / max(len(context), 1)
        uncertainty_phrases = ["i don't know", 'not enough information', 'cannot determine', 'not found', 'no information', 'does not contain', 'cannot answer', 'insufficient', 'unclear']
        uncertainty_flag = 1.0 if any((phrase in answer.lower() for phrase in uncertainty_phrases)) else 0.0
        return {'groundedness': round(groundedness, 3), 'relevance': round(relevance, 3), 'length_ratio': round(length_ratio, 3), 'uncertainty_flag': uncertainty_flag}

class OVLLMClient:

    def __init__(self, model_path: str, device: str='CPU'):
        self.model_path = model_path
        self.device = device
        self._pipeline = None
        logger.warning('OVLLMClient is a placeholder. Install openvino-genai and convert a model first.')

    def is_available(self) -> bool:
        try:
            import openvino_genai
            return True
        except ImportError:
            return False

    def generate(self, question: str, context: str='', **kwargs) -> str:
        return '[OVLLMClient] Not implemented. See src/llm/openvino_llm.py for the full placeholder.'
if __name__ == '__main__':
    '\n    Test script for the Ollama client.\n\n    Run:  python -m src.llm.ollama_client\n\n    Tests:\n      1. Server connectivity\n      2. Basic generation (no context)\n      3. RAG generation (with mock context)\n      4. Prompt template comparison\n      5. Temperature experiment\n      6. Streaming output\n      7. Answer quality metrics\n    '
    import textwrap

    def section(title: str) -> None:
        print(f"\n{'=' * 60}")
        print(f'  {title}')
        print(f"{'=' * 60}")
    client = OllamaClient()
    section('Test 1: Ollama Server Connectivity')
    available = client.is_available()
    print(f'  Server: {client.base_url}')
    print(f'  Model:  {client.model}')
    print(f"  Status: {('✓ Available' if available else '✗ Not available')}")
    if available:
        models = client.list_models()
        print(f'  Models: {models}')
    if not available:
        print('\n  ⚠ Ollama is not running. Start it with: ollama serve')
        print('  ⚠ Then pull the model: ollama pull mistral')
        print('  Remaining tests will be skipped.')
        sys.exit(0)
    section('Test 2: Basic Generation (no context)')
    answer = client.generate_without_context('What is Retrieval Augmented Generation (RAG) in one sentence?', temperature=0.3)
    print(f"\n  Q: What is RAG?\n  A: {textwrap.fill(answer, width=70, initial_indent='     ', subsequent_indent='     ')}")
    section('Test 3: RAG Generation (with context)')
    mock_context = '[Source 1 | score=0.85 | doc=invoice_001]\nInvoice Number: INV-2024-0042\nDate: March 15, 2024\nCompany: Acme Corporation\nTotal Amount: $1,250.00\nPayment Terms: Net 30\n\n[Source 2 | score=0.72 | doc=invoice_001]\nBill To: John Smith, 456 Oak Avenue, Springfield, IL 62704\nShip To: Same as billing address\n'
    question = 'What is the total amount on the invoice?'
    answer = client.generate(question=question, context=mock_context)
    print(f'\n  Q: {question}')
    print(f"  A: {textwrap.fill(answer, width=70, initial_indent='     ', subsequent_indent='     ')}")
    section('Test 4: Prompt Template Comparison')
    for tmpl_name, tmpl in PROMPT_TEMPLATES.items():
        print(f'\n  --- Template: {tmpl_name} ---')
        print(f"  Description: {tmpl['description']}")
        ans = client.generate(question='Who is the invoice billed to?', context=mock_context, template=tmpl_name)
        print(f"  Answer: {textwrap.fill(ans, width=65, initial_indent='', subsequent_indent='         ')}")
    section('Test 5: Temperature / Preset Experiment')
    for preset_name in ['precise', 'balanced', 'creative']:
        p = GENERATION_PRESETS[preset_name]
        ans = client.generate(question='What company issued this invoice?', context=mock_context, preset=preset_name)
        print(f"\n  Preset: {preset_name} (temp={p['temperature']}, top_p={p['top_p']})")
        print(f'  Answer: {ans[:200]}')
    section('Test 6: Streaming Response')
    print('\n  Streaming answer: ', end='', flush=True)
    full_answer = ''
    for chunk in client.generate_stream(question='What is the payment term?', context=mock_context):
        print(chunk, end='', flush=True)
        full_answer += chunk
    print('\n')
    section('Test 7: Answer Quality Metrics')
    metrics = OllamaClient.measure_answer_quality(answer=full_answer, context=mock_context, question='What is the payment term?')
    for metric, value in metrics.items():
        bar = '█' * int(value * 20)
        print(f'  {metric:<20}: {value:.3f}  {bar}')
    fake = 'The invoice was signed by Albert Einstein on the Moon.'
    fake_metrics = OllamaClient.measure_answer_quality(answer=fake, context=mock_context, question='Who signed the invoice?')
    print(f'\n  Hallucinated answer metrics:')
    for metric, value in fake_metrics.items():
        bar = '█' * int(value * 20)
        print(f'  {metric:<20}: {value:.3f}  {bar}')
    section('All Tests Complete!')
