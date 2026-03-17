/* ── overlays.js ──────────────────────────────────
   Hand-result and game-over overlay logic.
   Depends on: state.js, utils.js
─────────────────────────────────────────────── */

function showHandResult(rewards, showdown) {
  const entries = Object.entries(rewards).sort((a, b) => b[1] - a[1]);

  document.getElementById('result-street').textContent = `Hand ${handNumber} of ${totalHands}`;
  document.getElementById('result-list').innerHTML = entries.map(([name, delta]) => {
    const cls   = delta > 0 ? 'pos' : delta < 0 ? 'neg' : 'zero';
    const sign  = delta > 0 ? '+' : '';
    const cards = showdown && showdown[name];
    const cardsHTML = cards
      ? cards.map(c => cardHTML(c)).join('')
      : '<span style="color:var(--text-muted);font-family:\'DM Mono\',monospace;font-size:.7rem;">folded</span>';

    return `<div class="result-row" style="flex-direction:column;align-items:stretch;gap:8px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div class="result-name">${esc(name)}</div>
        <div class="result-delta ${cls}">${sign}${delta.toFixed(1)}</div>
      </div>
      <div style="display:flex;gap:6px;justify-content:center;">${cardsHTML}</div>
    </div>`;
  }).join('');

  document.getElementById('result-overlay').classList.add('show');

  const parts = entries.map(([n, d]) => {
    const cls  = d > 0 ? 'win' : d < 0 ? 'loss' : '';
    const sign = d > 0 ? '+' : '';
    return `${esc(n)}: <span class="${cls}">${sign}${d.toFixed(1)}</span>`;
  }).join(' · ');
  addLog(`🃏 Hand ${handNumber} — ${parts}`);
}

function closeResult() {
  document.getElementById('result-overlay').classList.remove('show');
}

function showGameOver(stacks) {
  closeResult();
  const sorted = Object.entries(stacks).sort((a, b) => b[1] - a[1]);
  document.getElementById('gameover-stacks').innerHTML = sorted.map(([n, s], i) => `
    <div class="result-row" style="margin-bottom:8px;">
      <div class="result-name">${i === 0 ? '🏆 ' : ''}${esc(n)}</div>
      <div class="result-delta ${i === 0 ? 'pos' : ''}">${fmt(s)} BB</div>
    </div>`
  ).join('');
  document.getElementById('gameover-overlay').classList.add('show');
  addLog(`🏁 <b>Game over!</b> Winner: <b>${sorted[0][0]}</b>`);
}
