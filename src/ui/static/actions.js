/* ── actions.js ───────────────────────────────────
   Action panel: check/call/fold/bet buttons,
   bet input, presets.
   Depends on: state.js, utils.js
─────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => disableActions());

function applyActionButtons(obs) {
  const a = obs.actions;
  const cc = document.getElementById("btn-checkcall");
  document.getElementById("action-panel").classList.add("your-turn");

  if (a.check) {
    cc.disabled = false;
    cc.className = "btn-action btn-check";
    cc.innerHTML = `<span class="btn-icon">✓</span><span class="btn-label">CHECK</span>`;
    cc.dataset.mode = "check";
  } else if (a.call) {
    cc.disabled = false;
    cc.className = "btn-action btn-call";
    cc.innerHTML = `<span class="btn-icon">↩</span><span class="btn-label">CALL</span><span class="btn-sub">${fmt(obs.bet_to_match)}</span>`;
    cc.dataset.mode = "call";
  } else {
    cc.disabled = true;
    cc.className = "btn-action btn-check";
    cc.innerHTML = `<span class="btn-icon">✓</span><span class="btn-label">CHECK</span>`;
    cc.dataset.mode = "check";
  }

  document.getElementById("btn-fold").disabled = !a.fold;
  document.getElementById("btn-raise").disabled = !a.bet;

  const lower = obs.bet_range.lower * BB_DOLLARS;
  const upper = obs.bet_range.upper * BB_DOLLARS;

  const input = document.getElementById("bet-input");
  input.min   = lower.toFixed(2);
  input.max   = upper.toFixed(2);
  input.step  = "0.25";
  input.value = lower.toFixed(2);
  _syncBetDisplay(lower);

  buildPresets(obs.bet_range.lower, obs.bet_range.upper, obs.pot);
  document.getElementById("bet-slider-wrap").classList.remove("open");
}

function disableActions() {
  ["btn-checkcall", "btn-fold", "btn-raise"].forEach(
    (id) => (document.getElementById(id).disabled = true),
  );
  document.getElementById("bet-slider-wrap").classList.remove("open");
  document.getElementById("action-panel").classList.remove("your-turn");
}

function doCheckCall() {
  doAction(document.getElementById("btn-checkcall").dataset.mode || "check");
}

function doAction(type) {
  if (!ws) return;
  disableActions();
  ws.send(JSON.stringify({ type: "action", action_type: type, bet_amount: 0 }));
  const banner = document.getElementById("turn-banner");
  banner.classList.remove("your-turn");
  banner.textContent = "Waiting…";
}

function toggleBetSlider() {
  document.getElementById("bet-slider-wrap").classList.toggle("open");
}

function _syncBetDisplay(dollars) {
  document.getElementById("bet-display-val").textContent =
    "$" + Number(dollars).toFixed(2);
}

function _clampBetInput() {
  const input   = document.getElementById("bet-input");
  const val     = parseFloat(input.value) || parseFloat(input.min);
  const clamped = Math.min(
    parseFloat(input.max),
    Math.max(parseFloat(input.min), val),
  );
  input.value = clamped.toFixed(2);
  return clamped;
}

// Aggiorna solo il display mentre si digita, senza clampare
function syncBetInput() {
  const dollars = parseFloat(document.getElementById("bet-input").value);
  if (!isNaN(dollars)) {
    _syncBetDisplay(dollars);
  }
}

function confirmBet() {
  if (!ws) return;
  // Clampa solo al momento della conferma, non durante la digitazione
  const dollars  = _clampBetInput();
  _syncBetDisplay(dollars);
  const bbAmount = dollars / BB_DOLLARS;
  disableActions();
  ws.send(
    JSON.stringify({ type: "action", action_type: "bet", bet_amount: bbAmount }),
  );
  const banner = document.getElementById("turn-banner");
  banner.classList.remove("your-turn");
  banner.textContent = "Waiting…";
}

function stepBet(direction) {
  const input   = document.getElementById("bet-input");
  const step    = parseFloat(input.step) || 0.25;
  const current = parseFloat(input.value) || parseFloat(input.min);
  const next    = Math.min(
    parseFloat(input.max),
    Math.max(parseFloat(input.min), current + direction * step),
  );
  input.value = next.toFixed(2);
  _syncBetDisplay(next);
}

function buildPresets(lowerBB, upperBB, potBB) {
  const lower = lowerBB * BB_DOLLARS;
  const upper = upperBB * BB_DOLLARS;
  const pot   = potBB   * BB_DOLLARS;

  const candidates = [
    { label: "½ pot",  val: pot * 0.5 },
    { label: "¾ pot",  val: pot * 0.75 },
    { label: "1× pot", val: pot },
    { label: "2× pot", val: pot * 2 },
    { label: "All-in", val: upper },
  ];

  document.getElementById("bet-presets").innerHTML = candidates
    .filter((p) => p.val >= lower - 0.01 && p.val <= upper + 0.01)
    .map((p) => {
      const v = Math.min(upper, Math.max(lower, Math.round(p.val * 4) / 4));
      return `<button class="btn-preset" onclick="setPreset(${v})">${p.label}<br><small>$${v.toFixed(2)}</small></button>`;
    })
    .join("");
}

function setPreset(dollarVal) {
  const input   = document.getElementById("bet-input");
  const clamped = Math.min(
    parseFloat(input.max),
    Math.max(parseFloat(input.min), dollarVal),
  );
  input.value = clamped.toFixed(2);
  _syncBetDisplay(clamped);
}