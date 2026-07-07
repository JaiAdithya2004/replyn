"""
generate_replies.py
-------------------
The Gen-AI response generator (evaluation version).

For every customer email in `dataset.json`, ask an LLM to draft a support reply.
The generation is GROUNDED IN THE DATASET via few-shot retrieval: for each email
we retrieve the most similar PAST emails and show their replies as examples, so
the model answers in the dataset's style and with its conventions.

LEAKAGE PROTECTION (important for a fair evaluation):
  When answering email X, we NEVER retrieve X's own gold reply as an example.
  Otherwise the model could just copy the answer and the evaluation would be
  meaningless. Each email is answered using OTHER emails only.

Writes results to `generated_replies.json`.

Run:  python generate_replies.py
      python generate_replies.py --no-grounding   # ablation: plain prompt, no few-shot
Uses keys from .env (OpenRouter -> Groq -> Gemini fallback). Mock replies if none.
"""

import argparse
import json
import math
import os
import re
from collections import Counter

import llm

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "dataset.json")
OUT_PATH = os.path.join(HERE, "generated_replies.json")

# ---- tiny TF-IDF retriever (same approach as the metric / product RAG) ------
_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP = set(
    "a an the and or but if then of to in on at for with from by is are was were "
    "be been being it its this that these those you your our we i he she they them "
    "as so not no do does did have has had will would can could should may might "
    "please thank thanks hi hello dear regards best".split()
)


def _tok(t):
    return [w for w in _WORD_RE.findall(t.lower()) if w not in _STOP and len(w) > 1]


def _build_idf(emails):
    docs = [set(_tok(e["customer_email"])) for e in emails]
    n = len(docs)
    df = Counter()
    for d in docs:
        for t in d:
            df[t] += 1
    return {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}, [Counter(_tok(e["customer_email"])) for e in emails]


def _cosine(tf_a, tf_b, idf):
    def vec(tf):
        return {t: (c / len(tf)) * idf.get(t, 1.0) for t, c in tf.items()}
    va, vb = vec(tf_a), vec(tf_b)
    dot = sum(va.get(t, 0.0) * vb.get(t, 0.0) for t in set(va) | set(vb))
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def retrieve_examples(idx, tfs, emails, idf, k=2):
    """Top-k most similar OTHER emails (never the email itself -> no leakage)."""
    q = tfs[idx]
    scored = []
    for j, tf in enumerate(tfs):
        if j == idx:
            continue  # LEAKAGE GUARD: skip the email we're answering
        scored.append((_cosine(q, tf, idf), j))
    scored.sort(reverse=True)
    return [emails[j] for s, j in scored[:k] if s > 0.03]


def build_messages(product_context, customer_email, examples):
    parts = [
        product_context,
        "Write ONE concise, professional email reply to the customer's message.\n"
        "- Open with a brief, warm acknowledgement.\n"
        "- Directly answer or resolve their request; be specific.\n"
        "- End with a clear next step or ONE specific question.\n"
        "- Output ONLY the email body. No subject line. Sign off as 'Support'.",
    ]
    if examples:
        ex_txt = "\n\n".join(
            f"[Similar past '{e['category']}' email]\n"
            f"Customer: {e['customer_email']}\nReply: {e['reference_reply']}"
            for e in examples
        )
        parts.append(
            "Here are similar past emails and how they were answered. Match this "
            "style and conventions (do not copy verbatim):\n\n" + ex_txt
        )
    system = "\n\n".join(parts)
    user = f'Customer email:\n"""\n{customer_email}\n"""'
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grounding", action="store_true",
                    help="Ablation: generate with a plain prompt, no few-shot examples.")
    args = ap.parse_args()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    product_context = data["product_context"]
    emails = data["emails"]

    grounded = not args.no_grounding
    idf, tfs = _build_idf(emails) if grounded else ({}, [])

    if not llm.has_api_key():
        print("WARNING: no API key in .env — generating MOCK replies.\n")

    mode = "few-shot grounded (dataset)" if grounded else "plain prompt (ablation)"
    print(f"Generating replies for {len(emails)} emails | mode: {mode} | "
          f"providers: {llm.active_providers() or ['mock']}\n")

    results = []
    for i, e in enumerate(emails, start=1):
        examples = retrieve_examples(i - 1, tfs, emails, idf, k=2) if grounded else []
        messages = build_messages(product_context, e["customer_email"], examples)
        reply = llm.chat(messages, temperature=0.3, max_tokens=600)
        results.append({
            "id": e["id"],
            "category": e["category"],
            "intent": e.get("intent", ""),
            "customer_email": e["customer_email"],
            "reference_reply": e["reference_reply"],
            "generated_reply": reply,
            "grounding_examples_used": len(examples),
        })
        print(f"[{i}/{len(emails)}] {e['id']} ({e['category']}) - "
              f"{len(reply)} chars, {len(examples)} example(s)")

    payload = {
        "model": llm.DEFAULT_MODEL,
        "grounding": "few-shot retrieval over dataset (leakage-guarded)" if grounded else "none",
        "count": len(results),
        "results": results,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(results)} generated replies to {OUT_PATH}")


if __name__ == "__main__":
    main()
