"""
evaluate.py
-----------
The accuracy / evaluation system. Scores each generated reply against the gold
reference reply, and reports per-response AND overall scores.

WHY THIS METRIC (short version — full argument in the README):
  A support email has no single "correct" string. Many different replies can be
  excellent. So token-overlap metrics (exact match, BLEU/ROUGE) are the WRONG
  tool — they punish valid paraphrases and reward copying. Quality for support
  email is really about: did it help, is it correct, is the tone right, is it
  complete? We therefore score on TWO complementary axes:

  1. SEMANTIC SIMILARITY to the gold reply (0-1)
     Cosine similarity of TF-IDF-weighted bag-of-words vectors. This is a free,
     dependency-light proxy for "did it cover the same substance as the human
     reply?" It rewards paraphrase over exact wording.

  2. LLM-AS-JUDGE RUBRIC (each 1-5, rescaled to 0-1)
     An LLM grades the reply on four dimensions a support lead actually cares
     about: helpfulness, correctness, tone/professionalism, completeness. The
     customer email + gold reply are given as context so the judge is grounded.

  OVERALL = weighted blend (default: 30% similarity, 70% rubric), because human
  judgement of the dimensions matters more than lexical closeness — similarity
  is a useful but blunt signal, so it gets the smaller weight.

Outputs:
  - Prints a per-response table and overall + per-category aggregates.
  - Writes full detail to `evaluation.json`.

Run:  python evaluate.py
"""

import json
import math
import os
import re
import time
from collections import Counter

import llm

HERE = os.path.dirname(os.path.abspath(__file__))
REPLIES_PATH = os.path.join(HERE, "generated_replies.json")
OUT_PATH = os.path.join(HERE, "evaluation.json")

# How much each axis counts toward the overall score.
SIM_WEIGHT = 0.30
RUBRIC_WEIGHT = 0.70

# Rubric dimensions and their relative weights within the rubric score.
RUBRIC_DIMS = {
    "helpfulness": 0.30,   # does it actually move the customer's issue forward?
    "correctness": 0.30,   # is the information right / consistent with the gold reply?
    "tone": 0.20,          # warm, professional, appropriate for support?
    "completeness": 0.20,  # does it cover what needs covering, with a next step?
}

_WORD_RE = re.compile(r"[a-z0-9']+")
# Very common words carry little meaning for similarity — drop them.
_STOP = set(
    "a an the and or but if then of to in on at for with from by is are was were "
    "be been being it its this that these those you your our we i he she they them "
    "as so not no do does did have has had will would can could should may might "
    "please thank thanks hi hello dear regards best".split()
)


def tokenize(text):
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1]


def tfidf_cosine(text_a, text_b):
    """
    TF-IDF cosine similarity over just these two documents. Self-contained (no
    corpus, no model download) so evaluation always runs. Returns 0..1.
    """
    tok_a, tok_b = tokenize(text_a), tokenize(text_b)
    if not tok_a or not tok_b:
        return 0.0
    tf_a, tf_b = Counter(tok_a), Counter(tok_b)
    vocab = set(tf_a) | set(tf_b)

    # Document frequency across the 2-doc "corpus" -> idf downweights words that
    # appear in both, emphasising the distinctive content words.
    def idf(term):
        df = (1 if term in tf_a else 0) + (1 if term in tf_b else 0)
        return math.log((1 + 2) / (1 + df)) + 1.0  # smoothed idf

    def vec(tf):
        return {t: (tf[t] / len(tf)) * idf(t) for t in tf}

    va, vb = vec(tf_a), vec(tf_b)
    dot = sum(va.get(t, 0.0) * vb.get(t, 0.0) for t in vocab)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


JUDGE_SYSTEM = (
    "You are a strict but fair customer-support quality reviewer. You grade a "
    "DRAFTED support reply against the original customer email and a reference "
    "reply written by a human agent. Score honestly; do not inflate."
)

JUDGE_TEMPLATE = """Grade the DRAFTED reply on each dimension from 1 (poor) to 5 (excellent).

Dimensions:
- helpfulness: Does it actually help resolve or advance the customer's request?
- correctness: Is the information accurate and consistent with the reference reply? Penalize contradictions or invented facts.
- tone: Is it warm, professional, and appropriate for customer support?
- completeness: Does it address what's needed and give a clear next step?

CUSTOMER EMAIL:
\"\"\"{customer}\"\"\"

REFERENCE REPLY (human gold answer, for grounding — the draft need not match it word-for-word):
\"\"\"{reference}\"\"\"

DRAFTED REPLY (the one you are grading):
\"\"\"{draft}\"\"\"

Respond with ONLY a JSON object, no markdown, in exactly this form:
{{"helpfulness": <1-5>, "correctness": <1-5>, "tone": <1-5>, "completeness": <1-5>, "rationale": "<one short sentence>"}}"""


def parse_judge_json(text):
    """Robustly pull the JSON object out of the judge's response."""
    # Strip code fences if the model added them.
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in judge output: {text[:200]}")
    return json.loads(match.group(0))


def llm_judge(customer, reference, draft):
    prompt = JUDGE_TEMPLATE.format(customer=customer, reference=reference, draft=draft)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    # Be resilient: if the judge call fails (e.g. all providers rate-limited) or
    # returns unparseable output, fall back to a neutral 3/5 and flag it, rather
    # than crashing the whole evaluation run.
    try:
        raw = llm.chat(messages, model=llm.DEFAULT_JUDGE_MODEL, temperature=0.0, max_tokens=300)
        data = parse_judge_json(raw)
        failed = False
    except Exception as e:  # noqa: BLE001
        print(f"    judge unavailable ({str(e)[:60]}) -> neutral 3/5 for this item")
        data, failed = {}, True

    scores = {}
    for dim in RUBRIC_DIMS:
        v = float(data.get(dim, 3))
        scores[dim] = max(1.0, min(5.0, v))  # clamp to 1..5
    scores["rationale"] = str(data.get("rationale", "")).strip()
    scores["judge_failed"] = failed
    return scores


def rubric_to_unit(rubric):
    """Weighted rubric score in 0..1 from the 1..5 dimension scores."""
    total = 0.0
    for dim, w in RUBRIC_DIMS.items():
        total += w * ((rubric[dim] - 1) / 4.0)  # map 1..5 -> 0..1
    return total


def grade_one(item):
    ref = item["reference_reply"]
    draft = item["generated_reply"]
    customer = item["customer_email"]

    similarity = tfidf_cosine(ref, draft)
    rubric = llm_judge(customer, ref, draft)
    # If the judge was rate-limited (neutral fallback), don't record a corrupted
    # score — raise so the caller skips this item and a resume retries it.
    if rubric.get("judge_failed"):
        raise RuntimeError("judge rate-limited; skip for resume")
    rubric_unit = rubric_to_unit(rubric)
    overall = SIM_WEIGHT * similarity + RUBRIC_WEIGHT * rubric_unit

    return {
        "id": item["id"],
        "category": item["category"],
        "similarity": round(similarity, 4),
        "rubric": {k: rubric[k] for k in RUBRIC_DIMS},
        "rubric_score": round(rubric_unit, 4),
        "overall": round(overall, 4),
        "rationale": rubric.get("rationale", ""),
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    with open(REPLIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    if not llm.has_api_key():
        print("WARNING: OPENROUTER_API_KEY not set — LLM-judge runs in MOCK mode "
              "(neutral 3/5 scores). Similarity is still real.\n")

    print(f"Evaluating {len(results)} replies "
          f"(similarity {int(SIM_WEIGHT*100)}% + rubric {int(RUBRIC_WEIGHT*100)}%)...\n")

    # Small pause between judge calls so we don't burst past free-tier rate
    # limits (tunable via EVAL_DELAY_SECONDS).
    delay = float(os.environ.get("EVAL_DELAY_SECONDS", "2.0"))

    # RESUME support: if a prior run wrote partial results to a progress file,
    # reuse them so a rate-limit interruption doesn't force starting over.
    progress_path = OUT_PATH + ".partial"
    done = {}
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                for g in json.load(f):
                    done[g["id"]] = g
            print(f"Resuming: {len(done)} replies already graded.\n")
        except Exception:  # noqa: BLE001
            done = {}

    graded = []
    skipped = 0
    for i, item in enumerate(results, start=1):
        if item["id"] in done:
            g = done[item["id"]]
        else:
            try:
                g = grade_one(item)
            except Exception:  # noqa: BLE001 — judge rate-limited; skip, retry on resume
                skipped += 1
                print(f"[{i}/{len(results)}] {item['id']:<11} skipped "
                      f"(rate-limited; re-run to complete it)")
                continue
            # persist progress after each new grade so we can resume
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(graded + [g], f)
        graded.append(g)
        r = g["rubric"]
        print(f"[{i}/{len(results)}] {g['id']:<11} {g['category']:<13} "
              f"sim={g['similarity']:.2f}  "
              f"help={r['helpfulness']:.0f} corr={r['correctness']:.0f} "
              f"tone={r['tone']:.0f} compl={r['completeness']:.0f}  "
              f"=> overall={g['overall']:.3f}")
        if item["id"] not in done and i < len(results):
            time.sleep(delay)

    if skipped:
        print(f"\n{skipped} item(s) were rate-limited and skipped. Re-run the same "
              f"command to finish them (already-graded items are cached).")

    # ---- Aggregates ----
    overall_mean = mean([g["overall"] for g in graded])
    sim_mean = mean([g["similarity"] for g in graded])
    rubric_mean = mean([g["rubric_score"] for g in graded])
    dim_means = {
        dim: mean([g["rubric"][dim] for g in graded]) for dim in RUBRIC_DIMS
    }

    by_cat = {}
    for g in graded:
        by_cat.setdefault(g["category"], []).append(g["overall"])
    cat_means = {c: round(mean(v), 4) for c, v in sorted(by_cat.items())}

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  Overall score        : {overall_mean:.3f} / 1.000  "
          f"({overall_mean*100:.1f}%)")
    print(f"  Semantic similarity  : {sim_mean:.3f}")
    print(f"  Rubric (LLM judge)   : {rubric_mean:.3f}")
    print("  Rubric dimensions (avg /5):")
    for dim, v in dim_means.items():
        print(f"      - {dim:<13}: {v:.2f}")
    print("  Per-category overall :")
    for c, v in cat_means.items():
        print(f"      - {c:<13}: {v:.3f}")

    payload = {
        "weights": {"similarity": SIM_WEIGHT, "rubric": RUBRIC_WEIGHT,
                    "rubric_dims": RUBRIC_DIMS},
        "overall_score": round(overall_mean, 4),
        "similarity_mean": round(sim_mean, 4),
        "rubric_mean": round(rubric_mean, 4),
        "rubric_dim_means": {k: round(v, 4) for k, v in dim_means.items()},
        "per_category": cat_means,
        "per_response": graded,
    }
    # Only write the FINAL file when every reply was graded (no rate-limit skips),
    # so we never overwrite good results with an incomplete run. Progress is kept
    # in the .partial file for a resume.
    if skipped or len(graded) < len(results):
        print(f"\n{len(graded)}/{len(results)} graded so far; {skipped} skipped. "
              f"Re-run to finish — progress is saved. (evaluation.json NOT overwritten.)")
        return
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print(f"\nWrote detailed evaluation to {OUT_PATH}")


if __name__ == "__main__":
    main()
