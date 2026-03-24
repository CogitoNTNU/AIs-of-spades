/* ── utils.js ─────────────────────────────────────
   Pure helpers with no side-effects.
─────────────────────────────────────────────── */

/**
 * Format a BB value as dollars: "$12.50"
 * All amounts from the server are in BB units; 1 BB = $5.
 */
function fmt(n) {
  if (n === undefined || n === null) return "—";
  const v = Number(n);
  if (isNaN(v)) return "—";
  const dollars = v * BB_DOLLARS;
  return "$" + dollars.toFixed(2);
}

/** Escape HTML special characters. */
function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/** Switch between the three top-level screens. */
function show(id) {
  ["screen-join", "screen-lobby", "screen-game", "screen-spectator"].forEach(
    (s) => (document.getElementById(s).style.display = "none"),
  );
  document.getElementById(id).style.display = "flex";
}

/** Append a timestamped entry to the hand-history log. */
function addLog(msg) {
  const log = document.getElementById("log");
  const ts = new Date().toLocaleTimeString("it", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML = `<span class="log-time">${ts}</span><span class="log-msg">${msg}</span>`;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}
