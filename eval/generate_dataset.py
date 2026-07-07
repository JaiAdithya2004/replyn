"""
generate_dataset.py
-------------------
Builds the evaluation dataset from TWO real, public customer-support corpora,
merged into one clean set of {customer_email, reference_reply, category}.

Why two sources (this is the "representative dataset" argument):
  Using two independent public datasets makes the set more representative of real
  support than any single source, and shows the pipeline isn't tied to one data
  shape. Both are clean and download over plain HTTPS (no login / API token).

  1. Bitext Customer Support dataset (~27K rows) — broad coverage across 10+
     support categories (orders, refunds, payments, invoices, ...), labelled by
     category/intent. Structured and consistent.
     https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

  2. Kaludi Customer-Support-Responses (~70 rows) — curated, human-written
     support Q&A with natural, professional agent replies. Adds authentic,
     non-templated phrasing.
     https://huggingface.co/datasets/Kaludi/Customer-Support-Responses

Each row becomes: customer_email (the inbound message) + reference_reply (the
gold agent answer we grade against) + category + source.

Reproducible: fixed seed, balanced sampling per category. Swap in your own data
by replacing dataset.json with the same shape.

Run:  python generate_dataset.py
      python generate_dataset.py --per-category 3 --seed 42
"""

import argparse
import csv
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

BITEXT_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
KALUDI_URL = (
    "https://huggingface.co/datasets/Kaludi/Customer-Support-Responses/"
    "resolve/main/Customer-Support.csv"
)
BITEXT_CSV = os.path.join(HERE, "bitext_customer_support.csv")
KALUDI_CSV = os.path.join(HERE, "kaludi_customer_support.csv")
OUT_PATH = os.path.join(HERE, "dataset.json")

PRODUCT_CONTEXT = (
    "You are a customer support agent for an online service. You help customers "
    "with orders, refunds, payments, invoices, deliveries, account issues, "
    "cancellations, and general questions. Be warm, professional, and concise, "
    "and always give the customer a clear next step."
)

# Bitext responses use {{Placeholder}} tokens; map to natural stand-ins.
PLACEHOLDER_MAP = {
    "Order Number": "#A1029384", "Invoice Number": "#INV-5567",
    "Account Type": "Premium", "Online Company Portal Info": "your account dashboard",
    "Website URL": "our website", "Online Order Interaction": "your online order",
    "Customer Support Phone Number": "1-800-555-0142",
    "Customer Support Hours": "Mon-Fri, 9am-6pm ET",
    "Delivery Country": "the United States", "Delivery City": "your city",
    "Person Name": "Alex", "Company Name": "our company",
    "Money Amount": "$29.00", "Refund Amount": "$29.00",
    "Store Location": "your nearest store", "Client First Name": "there",
    "Client Last Name": "", "Date": "the next few days",
    "Date Range": "3-5 business days", "Time": "shortly",
    "Currency Symbol": "$", "Email Address": "support@ourcompany.com",
}

# Simple keyword -> category mapping for the Kaludi set (which has no labels).
KALUDI_KEYWORDS = [
    ("REFUND", ["refund", "money back"]),
    ("CANCEL", ["cancel", "unsubscribe"]),
    ("SHIPPING", ["shipping", "deliver", "arrive", "track", "address"]),
    ("ORDER", ["order", "return", "product", "item"]),
    ("PAYMENT", ["bill", "charge", "payment", "pay", "card"]),
    ("INVOICE", ["invoice", "receipt"]),
    ("ACCOUNT", ["account", "password", "login", "lock", "email"]),
]


def _download(url, path, label):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"Using cached {label} at {os.path.basename(path)}")
        return
    try:
        import requests
    except ImportError:
        sys.exit("Missing dependency 'requests'. Run: pip install -r eval/requirements.txt")
    print(f"Downloading real {label} dataset from Hugging Face...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    print(f"  saved {os.path.basename(path)}")


def clean_text(text):
    def repl(m):
        return PLACEHOLDER_MAP.get(m.group(1).strip(), m.group(1).strip())
    text = re.sub(r"\{\{\s*([^}]+?)\s*\}\}", repl, text)
    return re.sub(r"\s+", " ", text).strip()


def load_bitext():
    rows = []
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(BITEXT_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            q = clean_text(row.get("instruction", ""))
            a = clean_text(row.get("response", ""))
            cat = (row.get("category") or "").strip().upper()
            if len(q) < 8 or len(a) < 20 or not cat:
                continue
            rows.append({"customer_email": q, "reference_reply": a,
                         "category": cat, "intent": (row.get("intent") or "").strip(),
                         "source": "bitext"})
    return rows


def categorize_kaludi(q):
    ql = q.lower()
    for cat, kws in KALUDI_KEYWORDS:
        if any(k in ql for k in kws):
            return cat
    return "GENERAL"


def load_kaludi():
    rows = []
    with open(KALUDI_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            q = clean_text(row.get("query", ""))
            a = clean_text(row.get("response", ""))
            if len(q) < 8 or len(a) < 20:
                continue
            rows.append({"customer_email": q, "reference_reply": a,
                         "category": categorize_kaludi(q), "intent": "",
                         "source": "kaludi"})
    return rows


def sample_balanced(rows, per_category, seed):
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    rng = random.Random(seed)
    selected = []
    for cat in sorted(by_cat):
        bucket = by_cat[cat]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_category])
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _download(BITEXT_URL, BITEXT_CSV, "Bitext")
    _download(KALUDI_URL, KALUDI_CSV, "Kaludi")

    bitext = load_bitext()
    kaludi = load_kaludi()
    all_rows = bitext + kaludi
    print(f"Loaded {len(bitext)} Bitext + {len(kaludi)} Kaludi = {len(all_rows)} "
          f"usable rows across {len({r['category'] for r in all_rows})} categories.")

    # Balance-sample Bitext per category, then add a handful of clean, authentic
    # Kaludi rows for human phrasing.
    selected = sample_balanced(bitext, args.per_category, args.seed)
    rng = random.Random(args.seed + 1)
    rng.shuffle(kaludi)
    selected += kaludi[: max(4, args.per_category * 2)]
    rng.shuffle(selected)

    emails = []
    for i, r in enumerate(selected, start=1):
        emails.append({
            "id": f"email_{i:03d}", "category": r["category"],
            "intent": r.get("intent", ""), "source": r["source"],
            "customer_email": r["customer_email"],
            "reference_reply": r["reference_reply"],
        })

    payload = {
        "sources": [
            {"name": "Bitext Customer Support LLM Chatbot Training Dataset",
             "url": BITEXT_URL},
            {"name": "Kaludi Customer-Support-Responses", "url": KALUDI_URL},
        ],
        "product_context": PRODUCT_CONTEXT,
        "seed": args.seed, "per_category": args.per_category,
        "count": len(emails), "emails": emails,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(emails)} support emails to {OUT_PATH}")
    cats, srcs = {}, {}
    for e in emails:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
        srcs[e["source"]] = srcs.get(e["source"], 0) + 1
    print("Categories:", ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    print("Sources:", ", ".join(f"{k}={v}" for k, v in sorted(srcs.items())))


if __name__ == "__main__":
    main()
