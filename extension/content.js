/*
 * content.js — injected into Gmail (https://mail.google.com/*)
 *
 * Grammarly-style experience:
 *   1. Watches Gmail for an open reply/compose box.
 *   2. Floats a small round "AI" icon in the bottom-right of that reply box.
 *   3. On click: grabs the email thread text, calls the backend (/suggest),
 *      and drops the AI draft straight into the reply box to review & send.
 *
 * The API key never lives here — the extension only calls your backend URL
 * (a hosted cloud server in production, or localhost while testing).
 */

(function () {
  "use strict";

  const BACKEND_DEFAULT = "http://127.0.0.1:8000";
  const ICON_CLASS = "ai-reply-fab";
  const MARK = "data-ai-reply-fab";

  function getSettings() {
    return new Promise((resolve) => {
      try {
        chrome.storage.sync.get(
          { backendUrl: BACKEND_DEFAULT, tone: "friendly" },
          (items) => resolve(items)
        );
      } catch (e) {
        resolve({ backendUrl: BACKEND_DEFAULT, tone: "friendly" });
      }
    });
  }

  // The editable reply body: a contenteditable div with role="textbox".
  function findReplyBox(scope) {
    return (scope || document).querySelector(
      'div[role="textbox"][contenteditable="true"]'
    );
  }

  // Read the visible email thread being replied to (page DOM, not Gmail API).
  function extractThreadText() {
    let parts = Array.from(document.querySelectorAll("div.a3s"))
      .map((n) => n.innerText.trim())
      .filter((t) => t.length > 0);
    if (parts.length === 0) {
      const conv = document.querySelector('div[role="main"]');
      if (conv) parts = [conv.innerText.trim()];
    }
    return parts.slice(-3).join("\n\n---\n\n").slice(0, 12000);
  }

  function insertIntoReply(box, text) {
    box.focus();
    box.innerHTML = "";
    for (const line of text.split("\n")) {
      const div = document.createElement("div");
      div.textContent = line.length ? line : "​"; // keep blank lines
      box.appendChild(div);
    }
    box.dispatchEvent(new Event("input", { bubbles: true }));
  }

  // ---- small status bubble shown above the icon while working -----------
  function setBubble(fab, msg, kind) {
    let bubble = fab.querySelector(".ai-reply-bubble");
    if (!msg) {
      if (bubble) bubble.remove();
      return;
    }
    if (!bubble) {
      bubble = document.createElement("div");
      bubble.className = "ai-reply-bubble";
      fab.appendChild(bubble);
    }
    bubble.textContent = msg;
    bubble.dataset.kind = kind || "info";
  }

  async function onClick(fab, scope) {
    const settings = await getSettings();
    const box = findReplyBox(scope) || findReplyBox(document);
    if (!box) return setBubble(fab, "No reply box found", "err"), autoClear(fab);

    const emailText = extractThreadText();
    if (!emailText) return setBubble(fab, "No email text found", "err"), autoClear(fab);

    fab.classList.add("loading");
    setBubble(fab, "Writing reply…", "info");

    try {
      const resp = await fetch(
        settings.backendUrl.replace(/\/$/, "") + "/suggest",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email_text: emailText, tone: settings.tone }),
        }
      );
      if (!resp.ok) {
        let detail = "HTTP " + resp.status;
        try {
          detail = (await resp.json()).detail || detail;
        } catch (_) {}
        throw new Error(detail);
      }
      const data = await resp.json();
      insertIntoReply(box, data.reply);
      setBubble(fab, "✓ Draft inserted", "ok");
    } catch (err) {
      console.error("[AI Reply]", err);
      const msg = String(err).includes("Failed to fetch")
        ? "Backend offline?"
        : err.message;
      setBubble(fab, msg, "err");
    } finally {
      fab.classList.remove("loading");
      autoClear(fab);
    }
  }

  function autoClear(fab) {
    setTimeout(() => setBubble(fab, "", "info"), 3000);
  }

  function makeFab(scope) {
    const fab = document.createElement("div");
    fab.className = ICON_CLASS;
    fab.setAttribute("role", "button");
    fab.setAttribute("tabindex", "0");
    fab.title = "Draft an AI reply based on this email";
    // "AI" wordmark inside the round icon.
    const label = document.createElement("span");
    label.className = "ai-reply-fab-label";
    label.textContent = "AI";
    fab.appendChild(label);
    fab.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      onClick(fab, scope);
    });
    return fab;
  }

  // Anchor a floating icon to each open reply box (bottom-right corner).
  function injectFabs() {
    const boxes = document.querySelectorAll(
      'div[role="textbox"][contenteditable="true"]'
    );
    boxes.forEach((box) => {
      // Find a positioned-ish container to attach the floating icon to.
      const scope =
        box.closest('div[role="dialog"]') ||
        box.closest("td") ||
        box.parentElement ||
        document.body;
      if (!scope || scope.getAttribute(MARK)) return;
      // Only attach to boxes that are actually visible.
      if (box.offsetParent === null) return;
      scope.setAttribute(MARK, "1");

      // Ensure the container can host an absolutely-positioned child.
      const pos = getComputedStyle(scope).position;
      if (pos === "static") scope.style.position = "relative";

      scope.appendChild(makeFab(scope));
    });
  }

  const observer = new MutationObserver(() => {
    window.requestAnimationFrame(injectFabs);
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  injectFabs();

  console.log("[AI Reply] content script loaded (floating icon).");
})();
