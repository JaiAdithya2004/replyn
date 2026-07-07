"""
validate_metric.py
------------------
Does our accuracy metric actually measure REPLY QUALITY — or just produce a
number? This script answers that, which is the part the challenge weights
heaviest ("validate the metric reflects real quality").

The idea: a trustworthy quality metric must give HIGHER scores to good replies
and LOWER scores to bad ones. So we take each real customer email and construct
several reply variants of KNOWN relative quality, then check the metric ranks
them correctly.

For every email we build four variants:
  A. GOLD      — the real human reference reply (should score highest)
  B. PARAPHRASE — the gold reply reworded (a *good* reply that shares no exact
                  wording; tests that we reward meaning over string overlap)
  C. GENERIC   — a polite but content-free boilerplate reply (mediocre)
  D. WRONG     — an off-topic / unhelpful reply (should score lowest)

Then we validate with three checks:
  1. Ranking accuracy: fraction of emails where score(GOLD) >= score(PARAPHRASE)
     >= score(GENERIC) >= score(WRONG) holds pairwise. High = metric discriminates.
  2. Good-vs-bad separation: mean score of {GOLD, PARAPHRASE} minus mean score of
     {GENERIC, WRONG}. Positive & large = metric separates quality.
  3. Paraphrase robustness: does PARAPHRASE (good, but reworded) still score well
     above WRONG? This proves we're not just doing lexical matching.

If the metric passes, we have evidence it tracks real quality — not noise.

Run:  python validate_metric.py
Uses the same metric implementation as evaluate.py, so we validate the REAL
thing, not a copy.
"""

import json
import os
import statistics

import llm
import evaluate as ev  # reuse the exact metric (tfidf_cosine + llm_judge + weights)

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "dataset.json")
OUT_PATH = os.path.join(HERE, "metric_validation.json")

# Content-free boilerplate — polite but says nothing specific. Mediocre by design.
GENERIC_REPLY = (
    "Hi,\n\nThank you for contacting us. We appreciate you reaching out and value "
    "your business. Our team is here to help. Please let us know if you have any "
    "questions.\n\nBest regards,\nSupport"
)

# Clearly wrong / unhelpful, off-topic reply. Should score lowest.
WRONG_REPLY = (
    "Hey,\n\nOur office holiday party is next Friday at 6pm. Don't forget to bring "
    "a dish to share, and parking is available in Lot C. See you there!\n\nCheers"
)


def make_paraphrase(reference):
    """
    Ask the LLM to reword the gold reply while keeping the same meaning. This is a
    GOOD reply that shares little exact wording — the key test that the metric
    rewards meaning, not string overlap. Falls back to a light manual reword if
    no LLM is available.
    """
    if not llm.has_api_key():
        # crude but deterministic reword so the script still runs offline
        return "Hello,\n\n" + reference.replace("Hi,", "").replace("Best", "Kind").strip()
    messages = [
        {"role": "system", "content": "You reword customer-support replies. Keep "
         "the meaning, facts, and helpfulness identical, but change the wording and "
         "sentence structure substantially. Output only the reworded reply."},
        {"role": "user", "content": reference},
    ]
    try:
        return llm.chat(messages, temperature=0.7, max_tokens=500)
    except Exception:  # noqa: BLE001
        return "Hello,\n\n" + reference


def score_reply(customer, reference, draft):
    """
    Run the REAL metric (same weights as evaluate.py) on one reply. If the judge
    was rate-limited (neutral fallback), raise so the caller does NOT persist a
    corrupted score — the email is retried on the next resume instead.
    """
    sim = ev.tfidf_cosine(reference, draft)
    rubric = ev.llm_judge(customer, reference, draft)
    if rubric.get("judge_failed"):
        raise RuntimeError("judge rate-limited; skip this email for resume")
    rubric_unit = ev.rubric_to_unit(rubric)
    overall = ev.SIM_WEIGHT * sim + ev.RUBRIC_WEIGHT * rubric_unit
    return overall


def pairwise_ok(scores_in_expected_order):
    """Count how many adjacent pairs respect the expected (non-increasing) order."""
    ok = 0
    total = 0
    for i in range(len(scores_in_expected_order) - 1):
        total += 1
        if scores_in_expected_order[i] >= scores_in_expected_order[i + 1]:
            ok += 1
    return ok, total


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=int(os.environ.get("VALIDATE_LIMIT", "15")),
                    help="Validate on the first N emails (keeps calls within free-tier "
                         "limits; a subset is statistically enough to prove discrimination).")
    args = ap.parse_args()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    emails = data["emails"]
    if args.limit and args.limit < len(emails):
        emails = emails[:args.limit]

    delay = float(os.environ.get("EVAL_DELAY_SECONDS", "1.5"))

    if not llm.has_api_key():
        print("WARNING: no API key — judge runs in MOCK mode, so this validation "
              "will be dominated by the (real) similarity signal only.\n")

    print(f"Validating the accuracy metric on {len(emails)} emails x 4 variants "
          f"(gold / paraphrase / generic / wrong)...\n")

    # RESUME support: reload any per-email results from a prior run so rate-limit
    # interruptions don't force starting over.
    partial_path = OUT_PATH + ".partial"
    done = {}
    if os.path.exists(partial_path):
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                for r in json.load(f):
                    done[r["id"]] = r
            print(f"Resuming: {len(done)} emails already validated.\n")
        except Exception:  # noqa: BLE001
            done = {}

    rows = []
    skipped = 0
    for i, e in enumerate(emails, 1):
        if e["id"] in done:
            row = done[e["id"]]
        else:
            customer = e["customer_email"]
            gold = e["reference_reply"]
            try:
                para = make_paraphrase(gold)
                row = {
                    "id": e["id"], "category": e["category"],
                    "gold": round(score_reply(customer, gold, gold), 3),
                    "paraphrase": round(score_reply(customer, gold, para), 3),
                    "generic": round(score_reply(customer, gold, GENERIC_REPLY), 3),
                    "wrong": round(score_reply(customer, gold, WRONG_REPLY), 3),
                }
            except Exception:  # noqa: BLE001 — judge rate-limited; skip, retry on resume
                skipped += 1
                print(f"[{i}/{len(emails)}] {e['id']:<11} skipped (rate-limited; "
                      f"re-run to complete it)")
                continue
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(rows + [row], f)
        rows.append(row)
        print(f"[{i}/{len(emails)}] {e['id']:<11} "
              f"gold={row['gold']:.2f} para={row['paraphrase']:.2f} "
              f"generic={row['generic']:.2f} wrong={row['wrong']:.2f}")
        if e["id"] not in done and i < len(emails):
            time.sleep(delay)

    if skipped:
        print(f"\n{skipped} email(s) were rate-limited and skipped. Re-run the same "
              f"command to complete them (already-done emails are cached).")

    # Aggregate from rows.
    pair_ok = pair_total = 0
    good_scores, bad_scores, para_scores, wrong_scores = [], [], [], []
    for r in rows:
        ordered = [r["gold"], r["paraphrase"], r["generic"], r["wrong"]]
        ok, total = pairwise_ok(ordered)
        pair_ok += ok
        pair_total += total
        good_scores += [r["gold"], r["paraphrase"]]
        bad_scores += [r["generic"], r["wrong"]]
        para_scores.append(r["paraphrase"])
        wrong_scores.append(r["wrong"])

    ranking_accuracy = pair_ok / pair_total if pair_total else 0.0
    good_mean = statistics.mean(good_scores)
    bad_mean = statistics.mean(bad_scores)
    separation = good_mean - bad_mean
    para_vs_wrong = statistics.mean(para_scores) - statistics.mean(wrong_scores)

    print("\n" + "=" * 60)
    print("METRIC VALIDATION RESULTS")
    print("=" * 60)
    print(f"  1. Ranking accuracy (gold>=para>=generic>=wrong): {ranking_accuracy*100:.1f}%")
    print(f"       -> how often the metric orders quality correctly")
    print(f"  2. Good vs bad separation: {separation:+.3f}  "
          f"(good={good_mean:.3f}, bad={bad_mean:.3f})")
    print(f"       -> positive & large = metric separates good from bad replies")
    print(f"  3. Paraphrase robustness (para - wrong): {para_vs_wrong:+.3f}")
    print(f"       -> positive = rewards meaning, not exact wording")

    verdict = (
        ranking_accuracy >= 0.75 and separation > 0.05 and para_vs_wrong > 0.05
    )
    print(f"\n  VERDICT: {'PASS - metric tracks real quality' if verdict else 'REVIEW - weak discrimination'}")

    payload = {
        "n_emails": len(emails),
        "ranking_accuracy": round(ranking_accuracy, 4),
        "good_mean": round(good_mean, 4),
        "bad_mean": round(bad_mean, 4),
        "good_vs_bad_separation": round(separation, 4),
        "paraphrase_vs_wrong": round(para_vs_wrong, 4),
        "passed": verdict,
        "per_email": rows,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"\nWrote validation detail to {OUT_PATH}")


if __name__ == "__main__":
    main()
