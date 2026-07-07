"""
api/suggest.py — Vercel serverless function.

POST /api/suggest   { "email_text": "...", "tone": "friendly" }
  -> { "reply": "...", "model": "...", "rag_examples_used": N }

This is the serverless equivalent of backend/server.py's /suggest endpoint. It
reuses the SAME logic — the LLM provider chain (eval/llm.py) and RAG retrieval
(backend/rag.py) — so the deployed product behaves like local dev.

Vercel runs each request through the BaseHTTPRequestHandler below. Keys are read
from Vercel's Environment Variables (set them in the Vercel dashboard, never in
code): GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# Make eval/ (llm.py) and backend/ (rag.py) importable from the function.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "backend"))

import llm  # noqa: E402
import rag  # noqa: E402

# Live/interactive: fail over between providers fast so a request never stalls.
llm.set_interactive(float(os.environ.get("MAX_RETRY_WAIT", "2")))

PRODUCT_CONTEXT = (
    "You are a customer support agent replying inside a shared team inbox. You "
    "help customers with orders, refunds, payments, invoices, deliveries, "
    "accounts, cancellations, and general questions."
)
TONE_HINTS = {
    "friendly": "Keep the tone warm and friendly.",
    "formal": "Keep the tone formal and professional.",
    "concise": "Be as concise as possible while still being complete.",
    "empathetic": "Lead with empathy; acknowledge any frustration first.",
}


def _clip(text, n):
    text = text.strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def _build_messages(email_text, tone):
    tone_hint = TONE_HINTS.get(tone.lower(), TONE_HINTS["friendly"])
    cases = rag.retrieve(email_text, k=2)
    parts = [
        PRODUCT_CONTEXT, tone_hint,
        "Write ONE email reply. Rules:\n"
        "- Get to the point fast: 3-6 sentences, no filler.\n"
        "- Answer or resolve the request directly and confidently.\n"
        "- End with one clear next step, or ONE specific question if needed.\n"
        "- Output ONLY the email body. No subject line. Sign off as 'Support'.",
    ]
    if cases:
        ex = "\n\n".join(
            f"[Example {i} — past '{c['category']}' case]\n"
            f"Customer: {_clip(c['customer'], 200)}\nOur reply: {_clip(c['reply'], 400)}"
            for i, c in enumerate(cases, 1)
        )
        parts.append("Similar past replies for grounding (do not copy verbatim):\n\n" + ex)
    messages = [
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": f'Reply to this customer email:\n"""\n{email_text.strip()}\n"""'},
    ]
    return messages, len(cases)


def _cors(handler):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        _cors(self)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:  # noqa: BLE001
            return self._json(400, {"detail": "Invalid JSON body."})

        email_text = (data.get("email_text") or "").strip()
        tone = data.get("tone", "friendly")
        if not email_text:
            return self._json(400, {"detail": "email_text is required."})
        if not llm.has_api_key():
            return self._json(503, {"detail": "No provider API key set on the server."})

        try:
            messages, n_cases = _build_messages(email_text, tone)
            reply = llm.chat(messages, temperature=0.3, max_tokens=400)
        except Exception as e:  # noqa: BLE001
            return self._json(502, {"detail": f"LLM call failed: {e}"})

        model = f"{llm.LAST_PROVIDER}:{llm.LAST_MODEL}" if llm.LAST_PROVIDER else llm.DEFAULT_MODEL
        return self._json(200, {"reply": reply, "model": model, "rag_examples_used": n_cases})
