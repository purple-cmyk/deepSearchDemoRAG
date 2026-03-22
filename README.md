# Project 2 : Demo
# Deep Search Assistant 

A local **multimodal RAG system** for document and video understanding, enabling semantic search and grounded question-answering using optimized inference.

---

##  Models

* **Embeddings:** BAAI/bge-small-en-v1.5
* **LLM:** Gemma 7B (via Ollama / OpenVINO)
* **Reranker:** BAAI/bge-reranker-base
* **Multimodal:** CLIP ViT-B/32

---

##  Key Components

* `LocalInferenceEngine` – LLM inference
* `KeywordRetriever` – hybrid retrieval (BM25 + dense)
* `RelevanceReranker` – cross-encoder ranking
* `DocumentPassageUnit` – structured retrieval output

---

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cli.py ingest --dataset docvqa
python cli.py ask "your question"
```

---

##  Architecture

Modular pipeline:

* **Core:** embeddings, retrieval, LLM
* **Processing:** ingestion & preprocessing
* **Utils:** indexing, monitoring, benchmarking

---

## Features

Multimodal RAG • Hybrid retrieval • Re-ranking • Hardware-aware inference • Edge-ready
