# Replyn — Gmail Extension + Backend Setup

A **Grammarly-style** Chrome extension that floats an **"AI"** icon inside your
Gmail reply box. Click it and an AI-drafted reply drops into the box, ready to
edit and send. The AI call happens on a **hosted backend** that safely holds the
API key — so end users just load the extension; no Python, no key on their side.

```
  Gmail (mail.google.com)
      │  content.js floats an "AI" icon in the reply box
      │  click → grabs the email thread text
      ▼
  Vercel serverless API (/api/suggest)   ← API keys live ONLY here
      │  1. RAG: retrieve the most similar PAST replies (on-brand grounding)
      │  2. Call the LLM (Groq → OpenRouter → Gemini fallback chain)
      ▼
  Draft is inserted into Gmail's reply box to review & send
```

**Why a backend?** A browser extension must never contain the API key — anyone
can unzip and read it. The key stays on the server. This is exactly how Grammarly
and Hiver are built: thin client in the page, secrets + AI on the server.

**Two things make the replies good, not generic:**

1. **RAG (Retrieval-Augmented Generation).** Before calling the model, the backend
   finds the most similar *past* support replies (from `eval/dataset.json`, which
   stands in for a team's resolved tickets) and feeds them to the model as
   examples. So the AI answers in the team's voice, with the team's facts — not a
   generic guess. This is the same feature a product like Hiver would ship, and it
   needs **no model training** — just retrieval at answer-time. See `backend/rag.py`.
2. **Provider fallback chain.** The backend tries **OpenRouter → Groq → Gemini**;
   if one is down or rate-limited, the next answers automatically. Reliable even
   on free tiers. See `eval/llm.py`.

---

## For the end user (once the backend is deployed)

1. Download and unzip the `extension/` folder.
2. Open `chrome://extensions`, turn on **Developer mode** (top-right).
3. Click **Load unpacked** and select the `extension/` folder.
4. Click the extension icon → paste the **backend URL** (your deployed cloud URL)
   → pick a **tone** → Save. *(This is the only setup step.)*
5. Open Gmail, open an email, hit **Reply**. A round **AI** icon floats in the
   bottom-right of the reply box. Click it → the draft appears. Edit & send.

That's the whole user experience: **unzip → load → click the AI icon.**

---

## For you — deploy once to Vercel (≈5 min, free)

The extension needs a backend URL. Deploy the whole repo to **Vercel** — it serves
both the landing page and the API (as serverless functions in `api/`).

1. Push this repo to GitHub.
2. Go to <https://vercel.com> → **Add New… → Project** → import the repo.
3. Vercel auto-detects `vercel.json`. Under **Settings → Environment Variables**, add
   at least one key (e.g. `GROQ_API_KEY = gsk_...` from console.groq.com/keys).
4. **Deploy.** You'll get a URL like `https://replyn.vercel.app`.
5. Put that URL in the extension popup as the **Backend URL**. Verify at
   `https://replyn.vercel.app/health`.

> **Cold starts:** Vercel spins the function down when idle, so the *first* request
> after a while takes ~1–3s to warm up, then it's fast. Fine for a demo.

**Local dev instead?** Run `python -m uvicorn backend.server:app --port 8000` and
set the Backend URL to `http://127.0.0.1:8000` — same logic, no deploy.

---

## Test locally first (before deploying)

You can run the whole thing on your machine before touching the cloud:

```powershell
# from the project root
py -m pip install -r backend/requirements.txt
$env:OPENROUTER_API_KEY = "sk-or-..."       # PowerShell (export ... on mac/linux)
py -m uvicorn backend.server:app --port 8000
```
Then in the extension popup, leave the Backend URL as `http://127.0.0.1:8000` and
click **Test backend connection** — you want `has_api_key: true`.

*(Icons are pre-generated. To regenerate: `py extension/make_icons.py`.)*

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Icon says **"Backend offline?"** | Backend URL wrong, or (local) server not running, or (cloud) still waking up. |
| **"…NO API key…"** | Set `OPENROUTER_API_KEY` on the server/host and redeploy or restart. |
| **429 / rate limited** | Free models are rate-limited; wait a moment (server also retries with backoff). |
| Icon doesn't appear | Refresh Gmail. It attaches when a reply box opens; Gmail's DOM changes occasionally. |

---

## MVP vs. full production

**This is a genuine working MVP** — hosted backend, floating icon, real drafts in
real Gmail. To be a shippable public product you'd add:

- **Official Gmail API + OAuth** instead of reading the page DOM (survives Gmail
  UI changes; proper read/send with user consent + Google review).
- **Per-user auth & billing** on the backend (so it's not your key for everyone),
  plus per-team "voice" and **RAG over past replies** for on-brand tone
  (no model training required).
- **Chrome Web Store** packaging & review (so users install in one click instead
  of loading unpacked).
- Logging, monitoring, and quota controls.
