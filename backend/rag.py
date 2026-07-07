"""
backend/rag.py
--------------
Lightweight RAG (Retrieval-Augmented Generation) over past support replies.

The idea (and why it matters):
  A generic prompt makes the AI write *plausible* replies. RAG makes it write
  replies grounded in how THIS team has actually answered similar emails before —
  same facts, same voice, same policies. No model training: we just retrieve the
  most similar past cases at answer-time and show them to the model as examples.

How it works here:
  - Knowledge base = the real support Q/A pairs from eval/dataset.json
    (customer_email -> reference_reply). In production this would be a team's
    own resolved tickets / help-docs.
  - Retrieval = TF-IDF bag-of-words cosine similarity (same free, dependency-free
    approach used by the evaluator). For a new email we score every KB entry and
    return the top-k most similar past cases.
  - Those cases are injected into the prompt as "here's how we handled similar
    emails" — so the reply stays on-brand and factually consistent.

This is intentionally simple (no vector DB, no embeddings service) so it runs
anywhere. The retrieval function is isolated, so swapping in neural embeddings
later is a one-function change.
"""

import json
import math
import os
import re
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_KB_PATH = os.path.join(os.path.dirname(_HERE), "eval", "dataset.json")

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP = set(
    "a an the and or but if then of to in on at for with from by is are was were "
    "be been being it its this that these those you your our we i he she they them "
    "as so not no do does did have has had will would can could should may might "
    "please thank thanks hi hello dear regards best".split()
)

# Loaded once at import: list of {customer, reply, category, tf} entries.
_KB = []
_IDF = {}


def _tokenize(text):
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _build_index():
    """Load the knowledge base and precompute IDF across all past emails."""
    global _KB, _IDF
    if not os.path.exists(_KB_PATH):
        return
    with open(_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for e in data.get("emails", []):
        toks = _tokenize(e["customer_email"])
        if not toks:
            continue
        _KB.append({
            "customer": e["customer_email"],
            "reply": e["reference_reply"],
            "category": e.get("category", ""),
            "tf": Counter(toks),
        })
        docs.append(set(toks))

    # IDF over the whole KB corpus so distinctive words matter more.
    n = len(docs)
    df = Counter()
    for d in docs:
        for t in d:
            df[t] += 1
    _IDF = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}


def _vec(tf):
    return {t: (c / len(tf)) * _IDF.get(t, 1.0) for t, c in tf.items()}


def _cosine(tf_a, tf_b):
    va, vb = _vec(tf_a), _vec(tf_b)
    dot = sum(va.get(t, 0.0) * vb.get(t, 0.0) for t in set(va) | set(vb))
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(email_text, k=3, min_score=0.05):
    """
    Return up to k past {customer, reply, category, score} cases most similar to
    email_text, best first. Empty list if the KB is unavailable or nothing is
    similar enough (so the caller can fall back to a plain reply).
    """
    if not _KB:
        return []
    q_tf = Counter(_tokenize(email_text))
    if not q_tf:
        return []
    scored = []
    for entry in _KB:
        s = _cosine(q_tf, entry["tf"])
        if s >= min_score:
            scored.append((s, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, entry in scored[:k]:
        out.append({
            "customer": entry["customer"],
            "reply": entry["reply"],
            "category": entry["category"],
            "score": round(s, 3),
        })
    return out


def kb_size():
    return len(_KB)


# Build the index at import time.
_build_index()
