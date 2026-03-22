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
DEFAULT_MODEL = 'gemma:7b'
DOCUMENT_AWARE_INSTRUCTION = 'Analyze the provided document excerpts and formulate a response that directly addresses the query using only information contained within those excerpts. If the excerpts do not contain sufficient information to formulate a complete response, indicate the limitation. Avoid speculation or information from external knowledge.\n\nDOCUMENT EXCERPTS:\n{context}\n'
QUERY_FRAMEWORK = 'Answer the following inquiry based on the document excerpts provided:\n\n{question}'
SOURCE_ATTRIBUTED_INSTRUCTION = 'You are tasked with analyzing document passages and generating responses that include source attribution. Reference each passage by its source identifier [Passage N]. Do not provide information that cannot be supported by the provided passages. If no passage addresses the query, state this explicitly.\n\nDOCUMENT PASSAGES:\n{context}\n'
ATTRIBUTED_QUERY_TEMPLATE = 'Query: {question}\n\nProvide a source-attributed response:'
COMPACT_EXTRACTION_INSTRUCTION = 'Extract the direct answer from the provided document section. Response should be concise, containing only the specific information requested. If not found in document, respond with "Information not available in source material."\n\nDocument Section:\n{context}\n'
COMPACT_QUERY_TEMPLATE = '{question}'
MINIMAL_CONTEXT_PROMPT = 'Using only the document context provided, answer:\n{context}\nQ: {question}\nA:'
RESPONSE_MODES: Dict[str, Dict[str, str]] = {'standard': {'system': DOCUMENT_AWARE_INSTRUCTION, 'user': QUERY_FRAMEWORK, 'description': 'Document-grounded response with uncertainty acknowledgment'}, 'attributed': {'system': SOURCE_ATTRIBUTED_INSTRUCTION, 'user': ATTRIBUTED_QUERY_TEMPLATE, 'description': 'Source-referenced responses with passage citations [Passage N]'}, 'extractive': {'system': COMPACT_EXTRACTION_INSTRUCTION, 'user': COMPACT_QUERY_TEMPLATE, 'description': 'Direct factual extraction: numbers, names, dates'}, 'minimal': {'system': '', 'user': MINIMAL_CONTEXT_PROMPT, 'description': 'Minimal intervention prompt for controlled generation'}}
INFERENCE_PROFILES: Dict[str, Dict] = {'conservative': {'temperature': 0.1, 'top_p': 0.4, 'description': 'Conservative inference for high-stakes factual queries'}, 'balanced': {'temperature': 0.35, 'top_p': 0.85, 'description': 'Balanced mode for general document analysis'}, 'exploratory': {'temperature': 0.7, 'top_p': 0.92, 'description': 'Exploratory mode for synthesis and explanation'}, 'maximum_variation': {'temperature': 0.95, 'top_p': 0.98, 'description': 'Maximum output diversity for creative analysis'}}

class LocalInferenceEngine:
    "
    Manages inference requests to a local document-analysis model.
    Provides multiple response generation modes for academic use cases.
    "

    def __init__(self, server_url: str=DEFAULT_OLLAMA_URL, model_identifier: str=DEFAULT_MODEL):
        self.server_url = server_url.rstrip('/')
        self.model_identifier = model_identifier

    def check_model_availability(self) -> bool:
        """
        Check if the specified model is available on the inference server.
        """
        try:
            url = f'{self.server_url}/api/tags'
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                available_models = [m['name'] for m in data.get('models', [])]
                model_found = any((self.model_identifier in m for m in available_models))
                if not model_found:
                    logger.warning("Model '%s' not available. Loaded: %s", self.model_identifier, available_models)
                return model_found
        except Exception as exc:
            logger.error('Server unavailable at %s: %s', self.server_url, exc)
            return False

    def get_available_models(self) -> List[str]:
        """
        Retrieve list of models available on the inference server.
        """
        try:
            url = f'{self.server_url}/api/tags'
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            return []

    @staticmethod
    def construct_document_prompt(query: str, document_context: str, response_mode: str='minimal') -> Dict[str, str]:
        """
        Construct prompt structure for document analysis task.
        """
        mode_config = RESPONSE_MODES.get(response_mode, RESPONSE_MODES['minimal'])
        if response_mode == 'minimal':
            if document_context:
                user_input = mode_config['user'].format(context=document_context, question=query)
            else:
                user_input = f'Q: {query}\nA:'
            return {'system': '', 'user': user_input}
        system_instruction = mode_config['system'].format(context=document_context) if document_context else ''
        user_input = mode_config['user'].format(question=query)
        return {'system': system_instruction, 'user': user_input}

    def compute_document_response(self, query: str, document_context: str='', temperature: float=0.3, top_p: float=0.9, max_tokens: int=1024, response_mode: str='standard', inference_profile: Optional[str]=None, debug: bool=False) -> str:
        """
        Generate response based on document analysis.
        """
        if inference_profile and inference_profile in INFERENCE_PROFILES:
            profile = INFERENCE_PROFILES[inference_profile]
            temperature = profile['temperature']
            top_p = profile['top_p']
            logger.info("Using profile '%s': temp=%.2f, top_p=%.2f", inference_profile, temperature, top_p)
        
        prompt_structure = self.construct_document_prompt(query, document_context, response_mode)
        system_instruction = prompt_structure['system']
        user_input = prompt_structure['user']
        
        if debug:
            print('\n' + '=' * 70)
            print('INFERENCE REQUEST DETAILS')
            print('=' * 70)
            print(f'Response Mode: {response_mode}')
            print(f'Temperature: {temperature:.2f}, Top-p: {top_p:.2f}')
            print(f'--- INSTRUCTION ---\n{system_instruction}')
            print(f'--- QUERY ---\n{user_input}')
            print('=' * 70 + '\n')
        
        request_payload = {'model': self.model_identifier, 'prompt': user_input, 'system': system_instruction, 'stream': False, 'options': {'temperature': temperature, 'top_p': top_p, 'num_predict': max_tokens}}
        
        try:
            endpoint = f'{self.server_url}/api/generate'
            request_body = json.dumps(request_payload).encode('utf-8')
            http_request = urllib.request.Request(endpoint, data=request_body, headers={'Content-Type': 'application/json'}, method='POST')
            
            start_timestamp = time.time()
            with urllib.request.urlopen(http_request, timeout=300) as response:
                response_data = json.loads(response.read().decode('utf-8'))
            
            elapsed_seconds = time.time() - start_timestamp
            response_text = response_data.get('response', '').strip()
            token_count = response_data.get('eval_count', 0)
            token_duration = response_data.get('eval_duration', 0)
            tokens_per_second = token_count / (token_duration / 1000000000.0) if token_duration else 0
            
            logger.info("Response generated (%d characters, %d tokens, %.2f tok/s, %.2fs)", len(response_text), token_count, tokens_per_second, elapsed_seconds)
            return response_text
            
        except urllib.error.URLError as exc:
            logger.error('Server request failed: %s', exc)
            return f'[ERROR] Unable to connect to inference server at {self.server_url}'
        except Exception as exc:
            logger.error('Response generation error: %s', exc)
            return f'[ERROR] Generation error: {exc}'

    def stream_document_response(self, query: str, document_context: str='', temperature: float=0.3, top_p: float=0.9, max_tokens: int=1024, response_mode: str='standard', inference_profile: Optional[str]=None) -> Generator[str, None, None]:
        """
        Stream response tokens as they are generated.
        """
        if inference_profile and inference_profile in INFERENCE_PROFILES:
            profile = INFERENCE_PROFILES[inference_profile]
            temperature = profile['temperature']
            top_p = profile['top_p']
        
        prompt_structure = self.construct_document_prompt(query, document_context, response_mode)
        request_payload = {'model': self.model_identifier, 'prompt': prompt_structure['user'], 'system': prompt_structure['system'], 'stream': True, 'options': {'temperature': temperature, 'top_p': top_p, 'num_predict': max_tokens}}
        
        try:
            endpoint = f'{self.server_url}/api/generate'
            request_body = json.dumps(request_payload).encode('utf-8')
            http_request = urllib.request.Request(endpoint, data=request_body, headers={'Content-Type': 'application/json'}, method='POST')
            
            with urllib.request.urlopen(http_request, timeout=300) as response:
                for line in response:
                    if not line.strip():
                        continue
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        token_content = chunk_data.get('response', '')
                        if token_content:
                            yield token_content
                        if chunk_data.get('done', False):
                            token_count = chunk_data.get('eval_count', 0)
                            token_duration = chunk_data.get('eval_duration', 0)
                            tokens_per_sec = token_count / (token_duration / 1000000000.0) if token_duration else 0
                            logger.info('Stream complete: %d tokens, %.2f tok/s', token_count, tokens_per_sec)
                            return
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as exc:
            yield f'\n[ERROR] Server unreachable at {self.server_url}: {exc}'
        except Exception as exc:
            yield f'\n[ERROR] Stream error: {exc}'

    def direct_inference(self, instruction: str, temperature: float=0.65, max_tokens: int=512) -> str:
        """
        Perform inference without document context (general-purpose generation).
        """
        request_payload = {'model': self.model_identifier, 'prompt': instruction, 'stream': False, 'options': {'temperature': temperature, 'num_predict': max_tokens}}
        
        try:
            endpoint = f'{self.server_url}/api/generate'
            request_body = json.dumps(request_payload).encode('utf-8')
            http_request = urllib.request.Request(endpoint, data=request_body, headers={'Content-Type': 'application/json'}, method='POST')
            
            with urllib.request.urlopen(http_request, timeout=300) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                return response_data.get('response', '').strip()
        except Exception as exc:
            logger.error('Direct inference failed: %s', exc)
            return f'[ERROR] Inference error: {exc}'

    @staticmethod
    def assess_response_coherence(response: str, document_context: str, original_query: str) -> Dict[str, float]:
        """
        Evaluate response quality against source documents and query relevance.
        Computes groundedness (coverage from documents) and relevance scores.
        """
        import re

        def extract_terms(text: str) -> set:
            return set(re.findall('\\w+', text.lower()))
        
        response_terms = extract_terms(response)
        document_terms = extract_terms(document_context)
        query_terms = extract_terms(original_query)
        
        # Common English stopwords
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
                       'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 
                       'during', 'before', 'after', 'above', 'below', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
                       'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 
                       'me', 'him', 'her', 'us', 'them'}
        
        response_content = response_terms - common_words
        document_content = document_terms - common_words
        query_content = query_terms - common_words
        
        # Groundedness: proportion of response terms found in documents
        if response_content:
            groundedness_score = len(response_content & document_content) / len(response_content)
        else:
            groundedness_score = 0.0
        
        # Relevance: proportion of query terms covered in response
        if query_content:
            relevance_score = len(query_content & response_content) / len(query_content)
        else:
            relevance_score = 0.0
        
        # Length ratio: response conciseness
        response_length_ratio = len(response) / max(len(document_context), 1)
        
        # Uncertainty detection
        uncertainty_indicators = ["i don't know", 'insufficient information', 'cannot determine', 
                                 'not found in', 'no information', 'documents do not contain', 
                                 'cannot answer', 'data unavailable', 'unclear from sources']
        uncertainty_present = 1.0 if any((phrase in response.lower() for phrase in uncertainty_indicators)) else 0.0
        
        return {
            'groundedness': round(groundedness_score, 3),
            'relevance': round(relevance_score, 3),
            'response_length_ratio': round(response_length_ratio, 3),
            'uncertainty_detected': uncertainty_present
        }

class OpenVINOInferenceEngine:
    """
    Interface for OpenVINO-optimized local inference.
    Placeholder for hardware-accelerated model execution.
    """

    def __init__(self, model_directory: str, hardware_device: str='CPU'):
        self.model_directory = model_directory
        self.hardware_device = hardware_device
        self._inference_pipeline = None
        logger.warning('OpenVINOInferenceEngine requires openvino-genai package and model conversion.')

    def check_hardware_support(self) -> bool:
        """
        Verify OpenVINO and hardware availability.
        """
        try:
            import openvino_genai
            return True
        except ImportError:
            return False

    def compute_response(self, query: str, document_context: str='', **kwargs) -> str:
        """
        Execute inference on OpenVINO-converted model.
        """
        return '[OpenVINOInferenceEngine] Requires openvino-genai. See documentation for model conversion.'
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
