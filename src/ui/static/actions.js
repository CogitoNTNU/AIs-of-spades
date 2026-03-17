/* ── actions.js ───────────────────────────────────
   Action panel: check/call/fold/bet buttons,
   bet slider, presets.
   Depends on: state.js, utils.js
─────────────────────────────────────────────── */

/** Enable and configure the action buttons for the current observation. */
function applyActionButtons(obs) {
  const a  = obs.actions;
  const cc = document.getElementById('btn-checkcall');
  document.getElementById('action-panel').classList.add('your-turn');

  if (a.check) {
    cc.disabled   = false;
    cc.className  = 'btn-action btn-check';
    cc.innerHTML  = `<span class="btn-icon">✓</span><span class="btn-label">CHECK</span>`;
    cc.dataset.mode = 'check';
  } else if (a.call) {
    cc.disabled   = false;
    cc.className  = 'btn-action btn-call';
    cc.innerHTML  = `<span class="btn-icon">↩</span><span class="btn-label">CALL</span><span class="btn-sub">${fmt(obs.bet_to_match)} BB</span>`;
    cc.dataset.mode = 'call';
  } else {
    cc.disabled   = true;
    cc.className  = 'btn-action btn-check';
    cc.innerHTML  = `<span class="btn-icon">✓</span><span class="btn-label">CHECK</span>`;
    cc.dataset.mode = 'check';
  }

  document.getElementById('btn-fold').disabled  = !a.fold;
  document.getElementById('btn-raise').disabled = !a.bet;

  // Slider setup
  const sl = document.getElementById('bet-slider');
  sl.min   = obs.bet_range.lower;
  sl.max   = obs.bet_range.upper;
  sl.value = obs.bet_range.lower;
  document.getElementById('bet-display-val').textContent = fmt(obs.bet_range.lower);
  buildPresets(obs.bet_range.lower, obs.bet_range.upper, obs.pot);
  document.getElementById('bet-slider-wrap').classList.remove('open');
}

/** Disable all action buttons (after acting). */
function disableActions() {
  ['btn-checkcall', 'btn-fold', 'btn-raise'].forEach(id =>
    document.getElementById(id).disabled = true
  );
  document.getElementById('bet-slider-wrap').classList.remove('open');
}

// ── Button handlers (called from HTML onclick) ──

function doCheckCall() {
  doAction(document.getElementById('btn-checkcall').dataset.mode || 'check');
}

function doAction(type) {
  if (!ws) return;
  disableActions();
  ws.send(JSON.stringify({ type: 'action', action_type: type, bet_amount: 0 }));
  const banner = document.getElementById('turn-banner');
  banner.classList.remove('your-turn');
  banner.textContent = 'Waiting…';
  document.getElementById('action-panel').classList.remove('your-turn');
  addLog(`↑ <b>${type.toUpperCase()}</b>`);
}

function toggleBetSlider() {
  document.getElementById('bet-slider-wrap').classList.toggle('open');
}

function syncBetSlider() {
  document.getElementById('bet-display-val').textContent =
    fmt(parseFloat(document.getElementById('bet-slider').value));
}

function confirmBet() {
  if (!ws) return;
  const amount = parseFloat(document.getElementById('bet-slider').value);
  disableActions();
  ws.send(JSON.stringify({ type: 'action', action_type: 'bet', bet_amount: amount }));
  const banner = document.getElementById('turn-banner');
  banner.classList.remove('your-turn');
  banner.textContent = 'Waiting…';
  document.getElementById('action-panel').classList.remove('your-turn');
  addLog(`↑ <b>BET</b> ${fmt(amount)} BB`);
}

// ── Bet presets ──

function buildPresets(lower, upper, pot) {
  const candidates = [
    { label: '½ pot',  val: pot * 0.5 },
    { label: '¾ pot',  val: pot * 0.75 },
    { label: '1× pot', val: pot },
    { label: '2× pot', val: pot * 2 },
    { label: 'All-in', val: upper },
  ];
  document.getElementById('bet-presets').innerHTML = candidates
    .filter(p => p.val >= lower && p.val <= upper + 0.01)
    .map(p => {
      const v = Math.min(upper, Math.max(lower, Math.round(p.val * 2) / 2));
      return `<button class="btn-preset" onclick="setPreset(${v})">${p.label}<br><small>${fmt(v)}</small></button>`;
    }).join('');
}

function setPreset(val) {
  const sl = document.getElementById('bet-slider');
  sl.value = Math.max(+sl.min, Math.min(+sl.max, val));
  document.getElementById('bet-display-val').textContent = fmt(+sl.value);
}
