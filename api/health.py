"""
api/health.py — Vercel serverless function.

GET /api/health -> liveness + which providers/RAG are configured.
Handy for confirming a deploy is live and keys are set.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "backend"))

import llm  # noqa: E402
import rag  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {
            "status": "ok",
            "has_api_key": llm.has_api_key(),
            "providers": llm.active_providers(),
            "groq_model": llm.GROQ_MODEL,
            "rag_enabled": rag.kb_size() > 0,
            "rag_knowledge_base_size": rag.kb_size(),
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
