/* ── actions.js ───────────────────────────────────
   Action panel: check/call/fold/bet buttons,
   bet slider, presets.
   Depends on: state.js, utils.js
─────────────────────────────────────────────── */

// Buttons start disabled — enabled ONLY by applyActionButtons() which is
// called exclusively from _applyYourTurn(). This prevents the player from
// acting when it's not their turn.
document.addEventListener("DOMContentLoaded", () => disableActions());

/** Enable and configure the action buttons for the current observation. */
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

  // Slider setup
  const sl = document.getElementById("bet-slider");
  sl.min = obs.bet_range.lower * BB_DOLLARS;
  sl.max = obs.bet_range.upper * BB_DOLLARS;
  sl.step = 0.25;
  sl.value = obs.bet_range.lower * BB_DOLLARS;
  document.getElementById("bet-display-val").textContent = fmt(
    obs.bet_range.lower,
  );
  buildPresets(obs.bet_range.lower, obs.bet_range.upper, obs.pot);
  document.getElementById("bet-slider-wrap").classList.remove("open");
}

/** Disable all action buttons (called immediately after sending any action,
 *  and on hand_result). Re-enabled only when your_turn arrives. */
function disableActions() {
  ["btn-checkcall", "btn-fold", "btn-raise"].forEach(
    (id) => (document.getElementById(id).disabled = true),
  );
  document.getElementById("bet-slider-wrap").classList.remove("open");
  document.getElementById("action-panel").classList.remove("your-turn");
}

// ── Button handlers (called from HTML onclick) ──

function doCheckCall() {
  doAction(document.getElementById("btn-checkcall").dataset.mode || "check");
}

function doAction(type) {
  if (!ws) return;
  disableActions(); // immediately lock buttons so double-click is impossible
  ws.send(JSON.stringify({ type: "action", action_type: type, bet_amount: 0 }));
  const banner = document.getElementById("turn-banner");
  banner.classList.remove("your-turn");
  banner.textContent = "Waiting…";
  addLog(`↑ <b>${type.toUpperCase()}</b>`);
}

function toggleBetSlider() {
  document.getElementById("bet-slider-wrap").classList.toggle("open");
}

function syncBetSlider() {
  // Slider is now in dollars — display directly
  const dollars = parseFloat(document.getElementById("bet-slider").value);
  document.getElementById("bet-display-val").textContent =
    "$" + dollars.toFixed(2);
}

function confirmBet() {
  if (!ws) return;
  // Convert dollars back to BB for the server
  const dollars = parseFloat(document.getElementById("bet-slider").value);
  const bbAmount = dollars / BB_DOLLARS;
  disableActions();
  ws.send(
    JSON.stringify({
      type: "action",
      action_type: "bet",
      bet_amount: bbAmount,
    }),
  );
  const banner = document.getElementById("turn-banner");
  banner.classList.remove("your-turn");
  banner.textContent = "Waiting…";
  addLog(`↑ <b>BET</b> $${dollars.toFixed(2)}`);
}

// ── Bet presets ──

function buildPresets(lowerBB, upperBB, potBB) {
  const lower = lowerBB * BB_DOLLARS;
  const upper = upperBB * BB_DOLLARS;
  const pot = potBB * BB_DOLLARS;

  const candidates = [
    { label: "½ pot", val: pot * 0.5 },
    { label: "¾ pot", val: pot * 0.75 },
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
  const sl = document.getElementById("bet-slider");
  sl.value = Math.max(+sl.min, Math.min(+sl.max, dollarVal));
  document.getElementById("bet-display-val").textContent =
    "$" + (+sl.value).toFixed(2);
}
