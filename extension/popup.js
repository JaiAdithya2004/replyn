/* popup.js — settings UI for the extension.
 * Stores the backend URL + preferred tone, and can ping /health so the user
 * knows the server is up and whether the API key is configured. */

const DEFAULTS = { backendUrl: "http://127.0.0.1:8000", tone: "friendly" };

const $ = (id) => document.getElementById(id);

function load() {
  chrome.storage.sync.get(DEFAULTS, (items) => {
    $("backendUrl").value = items.backendUrl;
    $("tone").value = items.tone;
  });
}

function save() {
  const backendUrl = $("backendUrl").value.trim() || DEFAULTS.backendUrl;
  const tone = $("tone").value;
  chrome.storage.sync.set({ backendUrl, tone }, () => {
    const saved = $("saved");
    saved.style.display = "block";
    setTimeout(() => (saved.style.display = "none"), 1800);
  });
}

function showStatus(msg, ok) {
  const el = $("status");
  el.textContent = msg;
  el.className = ok ? "ok" : "err";
}

async function testConnection() {
  const url = ($("backendUrl").value.trim() || DEFAULTS.backendUrl).replace(/\/$/, "");
  showStatus("Checking…", true);
  try {
    const resp = await fetch(url + "/health", { method: "GET" });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    if (data.has_api_key) {
      showStatus(`Connected ✓  Model: ${data.model}`, true);
    } else {
      showStatus("Connected, but server has NO API key set. Set OPENROUTER_API_KEY and restart it.", false);
    }
  } catch (e) {
    showStatus("Cannot reach backend. Is the server running? " + e.message, false);
  }
}

$("save").addEventListener("click", save);
$("test").addEventListener("click", testConnection);
document.addEventListener("DOMContentLoaded", load);
