"""
backend/server.py
-----------------
Production-style backend for the Gmail AI-reply extension.

Why a backend exists at all:
  A browser extension must NEVER hold the LLM API key — anyone could open
  devtools and steal it. So the extension talks only to THIS server, and the
  server is the only thing that knows the OpenRouter key. This is exactly how a
  real product (Hiver, etc.) is structured: thin client in Gmail, secrets and
  AI calls on the server.

What it exposes:
  GET  /health          -> liveness check
  POST /suggest         -> { "email_text": "...", "tone": "friendly" }
                           returns { "reply": "...", "model": "..." }

Run:
  pip install -r backend/requirements.txt
  # set your key first (never commit it):
  #   PowerShell:  $env:OPENROUTER_API_KEY="sk-or-..."
  #   bash:        export OPENROUTER_API_KEY="sk-or-..."
  uvicorn backend.server:app --reload --port 8000
  # or:  python backend/server.py
"""

import os
import sys
import time
import logging

# Make the eval/ client and this backend/ dir importable regardless of how the
# server is launched (uvicorn imports it as a package, so backend/ isn't
# automatically on sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # backend/ for rag.py


def _load_dotenv():
    """
    Minimal .env loader (no extra dependency). Reads KEY=VALUE lines from a .env
    file at the project root and sets them as environment variables — UNLESS the
    variable is already set in the real environment (real env always wins).

    This MUST run before `import llm`, because llm.py reads keys at import time.
    """
    env_path = os.path.join(_REPO_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # tolerate quotes
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import llm  # reused generator/LLM client from the eval project
import rag  # retrieval over past support replies (RAG)

# Interactive mode: fail over between providers fast so a live user in Gmail
# never waits on a rate-limited provider (default cap 2s per 429).
llm.set_interactive(float(os.environ.get("MAX_RETRY_WAIT", "2")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("email-agent")

app = FastAPI(title="AI Email Reply Backend", version="1.0.0")

# ---- CORS: allow the extension (and local tools) to call us --------------
# Gmail runs on https://mail.google.com. Chrome extensions send an
# "Origin: chrome-extension://<id>" header. We allow Gmail + any chrome
# extension origin. Tighten ALLOWED_ORIGINS in production to your exact IDs.
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://mail.google.com",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"chrome-extension://.*",  # any installed extension id
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Shared product/support context the model uses to stay on-tone. In a real
# deployment this would be per-team config loaded from a DB.
PRODUCT_CONTEXT = (
    "You are a professional customer support agent replying inside a shared team "
    "inbox. You help customers with orders, refunds, payments, invoices, "
    "deliveries, accounts, cancellations, and general questions."
)

TONE_HINTS = {
    "friendly": "Keep the tone warm and friendly.",
    "formal": "Keep the tone formal and professional.",
    "concise": "Be as concise as possible while still being complete.",
    "empathetic": "Lead with empathy; acknowledge any frustration first.",
}


# ---- Request / response models -------------------------------------------
class SuggestRequest(BaseModel):
    email_text: str = Field(..., min_length=1, max_length=12000,
                            description="The customer's email/thread text.")
    tone: str = Field("friendly", description="friendly | formal | concise | empathetic")


class SuggestResponse(BaseModel):
    reply: str
    model: str
    rag_examples_used: int = 0  # how many past cases were retrieved for grounding


# ---- Very small in-memory rate limiter -----------------------------------
# Protects a locally-run backend from a runaway loop hammering the free tier.
_RECENT = []
_RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "20"))


def _rate_ok():
    now = time.time()
    cutoff = now - 60
    while _RECENT and _RECENT[0] < cutoff:
        _RECENT.pop(0)
    if len(_RECENT) >= _RATE_LIMIT:
        return False
    _RECENT.append(now)
    return True


def _clip(text, n):
    text = text.strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def _format_examples(cases):
    """Render retrieved past cases as compact few-shot examples for the prompt.
    Examples are clipped to keep the prompt small (lower latency)."""
    blocks = []
    for i, c in enumerate(cases, 1):
        blocks.append(
            f"[Example {i} — past '{c['category']}' case]\n"
            f"Customer: {_clip(c['customer'], 200)}\n"
            f"Our reply: {_clip(c['reply'], 400)}"
        )
    return "\n\n".join(blocks)


def build_messages(email_text, tone):
    tone_hint = TONE_HINTS.get(tone.lower(), TONE_HINTS["friendly"])

    # RAG: pull the most similar past replies so the answer stays on-brand and
    # factually consistent. Cap at 2 examples (latency: fewer/shorter examples =>
    # smaller prompt => faster generation, with negligible quality loss).
    cases = rag.retrieve(email_text, k=2)

    system_parts = [
        PRODUCT_CONTEXT,
        tone_hint,
        # Tighter style guidance — crisp, confident, no over-explaining.
        "Write ONE email reply. Rules:\n"
        "- Get to the point fast: 3-6 sentences, no filler.\n"
        "- Answer or resolve the request directly and confidently.\n"
        "- End with one clear next step, or ONE specific question if you truly "
        "need more info — don't lecture the customer about their own email.\n"
        "- Match the style and facts of the past examples when relevant.\n"
        "- Output ONLY the email body. No subject line, no preamble, no meta "
        "commentary. Sign off as 'Support'.",
    ]
    if cases:
        system_parts.append(
            "Here is how our team has replied to similar emails before. Use them "
            "to stay on-brand and factually consistent (do not copy verbatim):\n\n"
            + _format_examples(cases)
        )

    system = "\n\n".join(system_parts)
    user = f'Now reply to this customer email:\n"""\n{email_text.strip()}\n"""'
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@app.get("/")
def root():
    # Friendly landing so visiting the base URL doesn't look like an error.
    return {
        "service": "AI Email Reply Backend",
        "status": "running",
        "has_api_key": llm.has_api_key(),
        "endpoints": {"health": "/health", "suggest": "POST /suggest", "docs": "/docs"},
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_api_key": llm.has_api_key(),
        "providers": llm.active_providers(),  # fallback order, e.g. ["openrouter","groq","gemini"]
        "model": llm.DEFAULT_MODEL,
        "gemini_model": llm.GEMINI_MODEL,
        "rag_enabled": rag.kb_size() > 0,
        "rag_knowledge_base_size": rag.kb_size(),
    }


@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    if not _rate_ok():
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Slow down.")

    if not llm.has_api_key():
        # Be explicit rather than silently returning a mock in a "real" server.
        raise HTTPException(
            status_code=503,
            detail="Server has no OPENROUTER_API_KEY set. Set it and restart.",
        )

    text = req.email_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="email_text is empty.")

    # Retrieve once here so we can both build the prompt and report the count.
    cases = rag.retrieve(text, k=3)

    try:
        messages = build_messages(text, req.tone)
        # 400 tokens is ample for a support reply and noticeably faster than 600.
        reply = llm.chat(messages, temperature=0.3, max_tokens=400)
    except Exception as e:  # noqa: BLE001
        log.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    log.info("Generated reply (%d chars) for %d-char email; %d RAG examples.",
             len(reply), len(text), len(cases))
    served_by = f"{llm.LAST_PROVIDER}:{llm.LAST_MODEL}" if llm.LAST_PROVIDER else llm.DEFAULT_MODEL
    return SuggestResponse(
        reply=reply, model=served_by, rag_examples_used=len(cases)
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend.server:app", host="127.0.0.1", port=port, reload=False)
