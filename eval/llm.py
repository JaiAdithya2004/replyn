"""
llm.py
------
Shared LLM client with an automatic PROVIDER FALLBACK CHAIN.

Order:
    1. OpenRouter  (primary)   -> one key, many free models (Llama, Gemini, ...)
    2. Google Gemini (fallback) -> used automatically if OpenRouter fails for ANY
                                   reason (error, timeout, rate-limit).

This gives reliability: if the primary provider is down or rate-limited, the
request transparently retries on the backup, so the extension/evaluator keep
working.

Keys are read ONLY from environment variables — never hard-coded, never
committed. Set whichever you have:

    OPENROUTER_API_KEY     get at https://openrouter.ai/keys
    GEMINI_API_KEY         get at https://aistudio.google.com/apikey

Optional model overrides:
    OPENROUTER_MODEL        default: meta-llama/llama-3.3-70b-instruct:free
    OPENROUTER_JUDGE_MODEL  default: same as OPENROUTER_MODEL
    GEMINI_MODEL            default: gemini-2.0-flash

If NO key is set at all, calls return a deterministic MOCK response so the
pipeline still runs end-to-end for a grader without keys.
"""

import json
import os
import time


def _load_dotenv():
    """
    Minimal .env loader (no dependency). Loads KEY=VALUE from the project-root
    .env so eval scripts get the same keys as the backend. Real environment
    variables always win over .env values.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
DEFAULT_JUDGE_MODEL = os.environ.get("OPENROUTER_JUDGE_MODEL", DEFAULT_MODEL)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Cap on how long any single provider waits on a 429 before failing over to the
# next provider in the chain. The INTERACTIVE product (Gmail) sets this low via
# set_interactive() so a live user never waits on a rate-limited provider — with
# multiple providers, failing over is faster than waiting. Offline eval keeps it
# high to ride out transient limits.
MAX_RETRY_WAIT = float(os.environ.get("MAX_RETRY_WAIT", "30"))


def set_interactive(max_wait=2.0):
    """Backend calls this so live requests fail over fast instead of waiting."""
    global MAX_RETRY_WAIT
    MAX_RETRY_WAIT = max_wait


# ----------------------------------------------------------------------------
# Key / capability checks
# ----------------------------------------------------------------------------
def has_openrouter():
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def has_groq():
    return bool(os.environ.get("GROQ_API_KEY"))


def has_gemini():
    return bool(os.environ.get("GEMINI_API_KEY"))


def has_api_key():
    """True if ANY provider is configured (kept for backwards compatibility)."""
    return has_openrouter() or has_groq() or has_gemini()


# active_providers() is defined below, after PROVIDER_ORDER is set.


# ----------------------------------------------------------------------------
# Mock (no keys configured)
# ----------------------------------------------------------------------------
def _mock_response(messages):
    blob = " ".join(m.get("content", "") for m in messages)
    if '"helpfulness"' in blob or ("helpfulness" in blob and "JSON" in blob):
        return json.dumps({
            "helpfulness": 3, "correctness": 3, "tone": 3, "completeness": 3,
            "rationale": "MOCK MODE: no API key set, returning neutral scores."
        })
    return ("Hi there,\n\nThanks for reaching out. (MOCK reply — set "
            "OPENROUTER_API_KEY or GEMINI_API_KEY for real replies.)\n\nBest,\nSupport")


# ----------------------------------------------------------------------------
# Provider: OpenRouter
# ----------------------------------------------------------------------------
def _retry_after(resp, attempt):
    """
    Work out how long to wait on a 429. Prefer the server's own guidance:
    the 'Retry-After' header, or OpenRouter's retry_after_seconds in the body.
    Fall back to exponential backoff. Capped so we never hang too long.
    """
    # 1) Standard Retry-After header (seconds).
    hdr = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if hdr:
        try:
            return min(float(hdr) + 1, MAX_RETRY_WAIT)
        except ValueError:
            pass
    # 2) OpenRouter puts retry_after_seconds inside the error metadata.
    try:
        meta = resp.json().get("error", {}).get("metadata", {})
        secs = meta.get("retry_after_seconds")
        if secs:
            return min(float(secs) + 1, MAX_RETRY_WAIT)
    except Exception:  # noqa: BLE001
        pass
    # 3) Exponential backoff fallback.
    return min(2 ** attempt + 2, MAX_RETRY_WAIT)


def _call_openrouter(messages, model, temperature, max_tokens, retries=4):
    import requests

    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/email-reply-eval",
        "X-Title": "AI Email Reply",
    }
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 data=json.dumps(payload), timeout=90)
            if resp.status_code == 429:
                wait = _retry_after(resp, attempt)
                # Don't sleep after the final attempt — let it fall through/fail.
                if attempt < retries - 1:
                    print(f"  [openrouter] rate limited, waiting {wait:.0f}s "
                          f"(attempt {attempt + 1}/{retries})...")
                    time.sleep(wait)
                last_err = RuntimeError("429 rate limited")
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"openrouter failed: {last_err}")


# ----------------------------------------------------------------------------
# Provider: Groq (OpenAI-compatible, fast, generous free tier)
# ----------------------------------------------------------------------------
def _call_groq(messages, temperature, max_tokens, retries=3):
    import requests

    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,  # Groq uses the OpenAI chat format directly
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(GROQ_URL, headers=headers,
                                 data=json.dumps(payload), timeout=90)
            if resp.status_code == 429:
                wait = _retry_after(resp, attempt)
                if attempt < retries - 1:
                    print(f"  [groq] rate limited, waiting {wait:.0f}s...")
                    time.sleep(wait)
                last_err = RuntimeError("429 rate limited")
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"groq failed: {last_err}")


# ----------------------------------------------------------------------------
# Provider: Google Gemini
# ----------------------------------------------------------------------------
def _messages_to_gemini(messages):
    """
    Gemini's REST API has no 'system' role. We fold the system prompt into the
    first user turn, and map assistant->model.
    """
    system_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    # Prepend system instructions to the first user message.
    if system_txt and contents:
        contents[0]["parts"][0]["text"] = system_txt + "\n\n" + contents[0]["parts"][0]["text"]
    elif system_txt:
        contents = [{"role": "user", "parts": [{"text": system_txt}]}]
    return contents


def _call_gemini(messages, temperature, max_tokens, retries=2):
    import requests

    url = GEMINI_URL.format(model=GEMINI_MODEL)
    headers = {"Content-Type": "application/json"}
    params = {"key": os.environ["GEMINI_API_KEY"]}
    payload = {
        "contents": _messages_to_gemini(messages),
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, params=params,
                                 data=json.dumps(payload), timeout=90)
            if resp.status_code == 429:
                wait = 2 ** attempt + 1
                print(f"  [gemini] rate limited, retrying in {wait}s...")
                time.sleep(wait)
                last_err = RuntimeError("429 rate limited")
                continue
            resp.raise_for_status()
            data = resp.json()
            # Extract text robustly — some models (thinking/multi-part) may not
            # include a "parts" list, or may block content.
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(
                    "gemini returned no candidates: " + json.dumps(data)[:200]
                )
            content = candidates[0].get("content", {}) or {}
            parts = content.get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                finish = candidates[0].get("finishReason", "unknown")
                raise RuntimeError(f"gemini returned empty text (finishReason={finish})")
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"gemini failed: {last_err}")


# ----------------------------------------------------------------------------
# Public entry point: try providers in order, fall back on ANY failure
# ----------------------------------------------------------------------------
# Provider order is configurable. Default puts Groq first because its free tier
# is the fastest and most reliable; OpenRouter's free pool is often rate-limited.
# Override with PROVIDER_ORDER, e.g. "openrouter,gemini".
_DEFAULT_ORDER = ["groq", "openrouter", "gemini"]
PROVIDER_ORDER = [
    p.strip().lower()
    for p in os.environ.get("PROVIDER_ORDER", ",".join(_DEFAULT_ORDER)).split(",")
    if p.strip()
]

_PROVIDERS = {
    "openrouter": (has_openrouter, lambda m, mo, t, mx: _call_openrouter(m, mo, t, mx)),
    "groq": (has_groq, lambda m, mo, t, mx: _call_groq(m, t, mx)),
    "gemini": (has_gemini, lambda m, mo, t, mx: _call_gemini(m, t, mx)),
}


def active_providers():
    """Configured providers, in the order they will be tried."""
    return [p for p in PROVIDER_ORDER if p in _PROVIDERS and _PROVIDERS[p][0]()]


def chat(messages, model=None, temperature=0.3, max_tokens=800):
    """
    Send a chat request through the provider fallback chain (PROVIDER_ORDER).
    On ANY failure of one provider, the next configured one is tried.
    Returns the assistant text. Mock response if no keys are set at all.
    """
    if not has_api_key():
        return _mock_response(messages)

    errors = []
    for name in PROVIDER_ORDER:
        entry = _PROVIDERS.get(name)
        if not entry:
            continue
        has_key, call = entry
        if not has_key():
            continue
        try:
            result = call(messages, model, temperature, max_tokens)
            # record which provider actually served this call (for reporting)
            global LAST_PROVIDER, LAST_MODEL
            LAST_PROVIDER = name
            LAST_MODEL = {"groq": GROQ_MODEL, "gemini": GEMINI_MODEL}.get(
                name, model or DEFAULT_MODEL)
            return result
        except Exception as e:  # noqa: BLE001
            print(f"  provider {name} failed -> trying next. ({e})")
            errors.append(f"{name}: {e}")

    raise RuntimeError("All providers failed: " + " | ".join(errors))


# Which provider/model served the most recent successful chat() call.
LAST_PROVIDER = None
LAST_MODEL = None
