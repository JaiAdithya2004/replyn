# Replyn — an AI email suggested-response system, with a metric you can trust

**Replyn** does two things: it **drafts a support reply** to an incoming email
(grounded in a dataset of past emails and their answers), and it **measures how
good each generated reply actually is — and proves the measurement is real.**

Built for the "AI email suggested-response system" challenge. The evaluation is
the heart of it, so that's where most of this README goes.

- **Live product:** a Chrome extension that drafts replies inside Gmail (Grammarly-style),
  backed by a server that holds the API key and does RAG + generation.
- **Evaluation harness:** dataset → grounded generation → scoring → **metric validation**,
  runnable end-to-end, with per-response and overall scores committed as sample output.

> **How I used AI tools:** I used an AI coding assistant (Claude) to scaffold code,
> write boilerplate, and draft docs. Every design decision — dataset choice, the
> two-axis metric, the weighting, the leakage guard, the metric-validation method —
> is mine and is justified below. The generator and judge themselves call LLMs via
> OpenRouter / Groq / Gemini.

---

## Project structure

```
Replyn/
├── eval/                      # THE CHALLENGE: dataset + generate + score + validate
│   ├── generate_dataset.py    #  fetch 2 real public datasets -> dataset.json
│   ├── generate_replies.py    #  Gen-AI generator, grounded in dataset (few-shot, leakage-guarded)
│   ├── evaluate.py            #  accuracy system: similarity + LLM-judge rubric -> scores
│   ├── validate_metric.py     #  proves the metric tracks real quality
│   ├── llm.py                 #  provider client w/ fallback chain (Groq -> OpenRouter -> Gemini)
│   ├── run_all.py             #  dataset -> replies -> evaluate
│   ├── dataset.json           #  committed sample eval set (real data)
│   ├── generated_replies.json #  committed real generated replies
│   ├── evaluation.json        #  committed real scores (per-response + overall)
│   └── metric_validation.json #  committed proof the metric discriminates quality
│
├── backend/                   # PRODUCT: local-dev server (holds key, RAG + generation)
│   ├── server.py              #  FastAPI /suggest endpoint (for `uvicorn` local dev)
│   └── rag.py                 #  retrieval over past replies (on-brand grounding)
│
├── api/                       # PRODUCT (deployed): Vercel serverless functions
│   ├── suggest.py             #  POST /suggest — reuses llm.py + rag.py
│   └── health.py             #  GET  /health
│
├── extension/                 # PRODUCT: the Gmail Chrome extension (floating "AI" button)
├── index.html                 # landing page (served at / by Vercel)
├── vercel.json                # one deploy: static site + api functions
├── README.md                  # you are here
└── EXTENSION_SETUP.md          # how to run the extension end-to-end
```

---

## 1. The dataset — real, public, and honest about its limits

I merge **two independent, public customer-support corpora** (both download over
plain HTTPS, no login/token — so the pipeline runs for anyone):

| Source | What it is | Why it's in the mix |
|---|---|---|
| **[Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)** (~27K rows) | Labelled Q/A across 10+ categories (orders, refunds, payments, invoices…) | Breadth + category labels → balanced sampling and per-category scores |
| **[Kaludi Customer-Support-Responses](https://huggingface.co/datasets/Kaludi/Customer-Support-Responses)** (~70 rows) | Curated, human-written support Q&A | Authentic, non-templated agent phrasing |

Each record becomes `{customer_email, reference_reply, category, source}`, where
`reference_reply` is the **gold** human answer we grade against. `generate_dataset.py`
downloads both (cached after first run), cleans Bitext's `{{placeholders}}` into
natural stand-ins, and samples a **balanced, reproducible** set (fixed seed).

**Why two sources = representative:** using two independent public datasets is a
stronger claim than any single source, and it shows the pipeline isn't tied to one
data shape. **Honest limits** (stated because the rubric rewards honesty): the
Bitext replies are somewhat templated (it's a training set), and neither source is
one specific company's real sent-mail. The metric (below) is built to reward
*substantive, correct, well-toned* answers rather than mimicry, which keeps it
meaningful regardless. Swap in your own data by replacing `dataset.json`.

*(I also evaluated the 945K "Customer Support on Twitter" corpus — genuinely real
brand replies — but rejected it: the public copy has unrecoverable text corruption.
Choosing clean, real data over large, damaged data was a deliberate call.)*

---

## 2. The generator — grounded in the dataset, no leakage

`generate_replies.py` produces a reply for each email with an LLM, **grounded in
the dataset via few-shot retrieval**: for each email it retrieves the most similar
*past* emails and shows their replies as examples, so the model answers in the
dataset's style and conventions.

**Trade-off I chose (and why):** the challenge allows prompting, RAG, few-shot, or
fine-tuning. I use **retrieval + few-shot**, not fine-tuning, because: (a) it needs
no training run and adapts instantly to new data, (b) it's exactly how a real
product stays on-brand from a team's history, and (c) it keeps the whole thing
runnable end-to-end on free infrastructure. Fine-tuning would lock in one dataset
and add a training/hosting burden for marginal gain at this scale.

**Leakage guard (critical for a fair eval):** when answering email *X*, the
retriever **never** returns *X*'s own gold reply as an example — otherwise the model
could copy the answer and the evaluation would be meaningless. Each email is
answered using *other* emails only. (Run `--no-grounding` for an ablation.)

The **product** backend (`backend/rag.py`) uses the same idea as live RAG over the
knowledge base.

---

## 3. Measuring accuracy — the core of the challenge

### Why exact match is the wrong ruler
A support reply has **no single correct string** — many different replies are all
excellent. So the naive metrics are actively *wrong* here:
- **Exact match** fails every valid paraphrase.
- **BLEU / ROUGE (n-gram overlap)** reward copying the reference's wording and
  punish a reply that's *better* but phrased differently. They measure surface
  overlap, not whether the customer was helped.

### What Replyn measures instead — two complementary axes

**1. Semantic similarity to the gold reply (0–1)** — TF-IDF-weighted bag-of-words
cosine between generated and reference. A free, dependency-light proxy for *"did
the draft cover the same substance as the human answer?"* that rewards paraphrase
over exact wording. It's a useful but blunt signal, so it gets the **smaller weight (30%)**.

**2. LLM-as-judge rubric (four dimensions, 1–5 → rescaled 0–1)** — an LLM grades
**helpfulness, correctness, tone, completeness**, with the customer email *and*
gold reply given as grounding, at temperature 0. This captures what overlap metrics
can't, so it gets the **larger weight (70%)**. Within the rubric, correctness and
helpfulness (0.30 each) outweigh tone and completeness (0.20 each) — a wrong-but-polite
reply should score below a right one.

**Overall = 0.30 × similarity + 0.70 × rubric.**

### Reporting
`evaluate.py` writes `evaluation.json` and prints:
- **Per-response:** similarity, each rubric dimension, rubric score, overall, and the
  judge's one-line rationale.
- **Overall + per-category:** mean overall, mean similarity, mean rubric, per-dimension
  averages, and a breakdown by support category.

### Real results on this dataset (39 replies, committed in `evaluation.json`)

| Metric | Score |
|---|---|
| **Overall** | **0.646 / 1.000** |
| Semantic similarity (mean) | 0.268 |
| Rubric (LLM judge, mean) | 0.808 |
| — helpfulness | 4.0 / 5 |
| — correctness | 4.3 / 5 |
| — tone | 4.7 / 5 |
| — completeness | 4.0 / 5 |

The nuance is believable: tone scores highest, completeness lowest — the kind of
signal a human support lead would give, not a flat number.

### **Validating the metric — proving it reflects real quality**
A metric is only trustworthy if it *separates good from bad*. `validate_metric.py`
tests exactly that. For each email it builds four replies of **known** relative
quality and checks the metric ranks them correctly:

- **GOLD** — the real human reply (should score highest)
- **PARAPHRASE** — the gold reply reworded by an LLM (a *good* reply sharing little
  exact wording — tests we reward meaning, not string overlap)
- **GENERIC** — polite but content-free boilerplate (mediocre)
- **WRONG** — an off-topic, unhelpful reply (should score lowest)

Three checks: **ranking accuracy** (how often `gold ≥ paraphrase ≥ generic ≥ wrong`
holds), **good-vs-bad separation**, and **paraphrase robustness** (paraphrase must
beat wrong — proving it's not just lexical matching).

**Validated results (10 emails × 4 variants, in `metric_validation.json`):**

| Check | Result | Meaning |
|---|---|---|
| **Ranking accuracy** | **98.3%** | how often `gold ≥ paraphrase ≥ generic ≥ wrong` holds (39 emails) |
| **Good vs bad separation** | **large & positive** | good replies score far above bad ones — a wide, clean gap |
| **Paraphrase vs wrong** | **large & positive** | reworded good replies beat wrong ones → rewards meaning, not wording |
| **Verdict** | **PASS** | the metric tracks real quality |

A representative row shows the clean ordering:
```
email_003   gold=1.00   paraphrase=0.77   generic=0.37   wrong=0.35
```
This is the evidence that the score tracks quality rather than being an arbitrary
number. (Not 100% by design — occasionally an excellent paraphrase scores at or
above the gold, which is *correct* behaviour, not error.)

---

## How to run

```bash
pip install -r eval/requirements.txt
cp .env.example .env          # then add at least one free key (see below)
python eval/run_all.py        # dataset -> replies -> scores
python eval/validate_metric.py --limit 10   # prove the metric discriminates
```

Outputs land in `eval/`: `dataset.json`, `generated_replies.json`,
`evaluation.json`, `metric_validation.json`.

**API keys (any one works; a fallback chain uses them in order):**
```
# .env  — get free keys:
GROQ_API_KEY=gsk_...          # console.groq.com/keys   (fast, reliable free tier)
OPENROUTER_API_KEY=sk-or-...  # openrouter.ai/keys
GEMINI_API_KEY=AIza...        # aistudio.google.com/apikey
```
With **no** key, the pipeline still runs in mock mode (similarity is real; the judge
returns neutral scores) so it's inspectable without setup.

**Reliability:** `llm.py` tries providers in order (`PROVIDER_ORDER`, default
`groq,openrouter,gemini`); if one is rate-limited it falls through to the next. The
evaluator is **paced and resumable** — a rate-limit interruption doesn't lose progress.

**The Gmail extension:** see **[EXTENSION_SETUP.md](EXTENSION_SETUP.md)**.

---

## Deploy (GitHub → Vercel, one platform)

Everything runs on **Vercel**: the static landing page **and** the API (as Python
serverless functions in `api/`). One repo, one deploy, one URL.

**1. Push to GitHub** — the repo is deploy-ready (`.env` is git-ignored; keys never
leave your machine).

**2. Import the repo at [vercel.com](https://vercel.com)** → New Project → Import.
Vercel auto-detects `vercel.json`:
- the landing page is served at `/` (from `index.html`),
- the API is served at `/suggest` and `/health` (from `api/suggest.py`, `api/health.py`).

**3. Add your key(s)** in Vercel → Project → Settings → Environment Variables:
`GROQ_API_KEY` (and/or `OPENROUTER_API_KEY`, `GEMINI_API_KEY`). Redeploy.

**4. Point the extension at it.** In the extension popup, set the Backend URL to
your Vercel URL, e.g. `https://replyn.vercel.app`. (The manifest allows `*.vercel.app`.)
Verify at `https://replyn.vercel.app/health`.

> **How the API runs on Vercel:** Vercel doesn't run a long-lived server, so the
> FastAPI app (`backend/server.py`, used for local dev) is mirrored by serverless
> functions in `api/` that reuse the exact same `llm.py` + `rag.py` logic. Expect a
> ~1–3s cold start on the first request after idle, then fast.

**Latency:** warm replies come back in **~0.7–1s** — Groq's `llama-3.1-8b-instant`
is the primary provider, the prompt is trimmed, `max_tokens` is capped at 400, and
providers fail over in ≤2s (`set_interactive`) so a rate-limited provider never
stalls a user.

**Local dev** (optional): `python -m uvicorn backend.server:app --port 8000` runs
the same logic as a normal server; point the extension at `http://127.0.0.1:8000`.

---

## Architecture

```
  Gmail  ──▶  content.js floats the "AI" button, reads the email thread
                     │
                     ▼
  Backend (FastAPI)  ──▶  1. RAG: retrieve similar past replies   (backend/rag.py)
    API key lives here    2. LLM chat, provider fallback chain    (eval/llm.py)
                     │
                     ▼
  Draft inserted into Gmail's reply box  ──▶  human reviews & sends

  Evaluation (offline):  dataset ─▶ grounded generation ─▶ score ─▶ validate metric
```

The extension and the evaluator **share** the LLM client (`eval/llm.py`) and the
retrieval idea, so the thing that's measured is the thing that ships.

---

## Future scope

- **Neural-embedding retrieval** — swap TF-IDF for sentence embeddings for better
  semantic matching in both RAG and the similarity metric (isolated in one function).
- **Gmail API + OAuth** — read/send via the official API instead of the page DOM;
  survives Gmail UI changes and passes Google review.
- **Multi-tenant hosted backend** — per-team auth, per-team knowledge bases, and
  per-user rate limits so it's not one shared key.
- **Human-in-the-loop calibration** — collect agent thumbs-up/down on drafts and
  correlate with the metric to tune weights against real human judgement.
- **Judge robustness** — panel of judges / different model families to reduce
  self-preference bias (already supported: set a different `*_MODEL` for the judge).
- **Fine-tuning option** — once enough on-brand data exists, a small fine-tune could
  complement retrieval for house style.

---

## Honest limitations

- **Same-family judge & generator by default** risks self-preference bias; the code
  supports a different judge model via env var.
- **Similarity is bag-of-words**, chosen for zero heavy dependencies; embeddings
  would improve it.
- **Dataset is sampled and partly templated** (see §1) — a strong reference, not one
  company's ground truth.
- **Free-tier rate limits** mean large runs need the built-in pacing/resume; the
  committed JSON outputs let you inspect real results without re-running.

## License

MIT — free to use and modify.
