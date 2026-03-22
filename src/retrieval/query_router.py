import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryPlan:
    top_k: int
    use_preprocessing: bool
    use_hybrid: bool
    dense_recall_mult: int
    route_label: str
    prefer_metadata: bool


_WORD_RE = re.compile('\\w+', re.UNICODE)
_YEAR_RE = re.compile('\\b(19|20)\\d{2}\\b')
_QUESTION_START = re.compile('^(what|who|when|where|why|how|which|is|are|can|could|would|should|do|does|did)\\b', re.I)


def analyze_query(query: str, base_top_k: int, bm25_available: bool=False) -> QueryPlan:
    q = query.strip()
    if not q:
        return QueryPlan(top_k=base_top_k, use_preprocessing=False, use_hybrid=False, dense_recall_mult=2, route_label='empty', prefer_metadata=False)
    words = _WORD_RE.findall(q)
    n_words = len(words)
    has_qmark = '?' in q
    is_question = has_qmark or bool(_QUESTION_START.match(q))
    has_year = bool(_YEAR_RE.search(q))
    if n_words <= 4 and (not is_question):
        label = 'keyword'
        top_k = min(base_top_k * 2, 24)
        preprocess = True
        hybrid = bm25_available
        dense_mult = 4
    elif n_words > 40 or len(q) > 280:
        label = 'long_context'
        top_k = min(int(base_top_k * 1.5) + 2, 30)
        preprocess = False
        hybrid = bm25_available and n_words > 60
        dense_mult = 3
    elif is_question:
        label = 'question'
        top_k = base_top_k
        preprocess = True
        hybrid = bm25_available
        dense_mult = 3
    else:
        label = 'semantic'
        top_k = base_top_k
        preprocess = False
        hybrid = False
        dense_mult = 2
    return QueryPlan(top_k=top_k, use_preprocessing=preprocess, use_hybrid=hybrid, dense_recall_mult=dense_mult, route_label=label, prefer_metadata=has_year)
