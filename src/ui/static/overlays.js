/* ── overlays.js ──────────────────────────────────
   Hand-result and game-over overlay logic.
   Depends on: state.js, utils.js
─────────────────────────────────────────────── */

function showHandResult(rewards, showdown, players, tableCards) {
  const entries = Object.entries(rewards).sort((a, b) => b[1] - a[1]);

  document.getElementById("result-street").textContent =
    `Hand ${handNumber} of ${totalHands}`;

  // Community cards in cima
  const communityHTML =
    tableCards && tableCards.length
      ? `<div style="margin-bottom:16px;">
        <div style="font-family:'DM Mono',monospace;font-size:.6rem;letter-spacing:3px;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px;">Board</div>
        <div style="display:flex;gap:6px;justify-content:center;">
          ${tableCards.map((c) => cardHTML(c)).join("")}
        </div>
      </div>`
      : "";

  // Righe per ogni giocatore
  const playersHTML = entries
    .map(([name, delta]) => {
      const cls = delta > 0 ? "pos" : delta < 0 ? "neg" : "zero";
      const sign = delta > 0 ? "+" : "";
      const cards = showdown && showdown[name];
      const playerInfo = players && players.find((p) => p.name === name);

      const cardsHTML = cards
        ? cards.map((c) => cardHTML(c)).join("")
        : "<span style=\"color:var(--text-muted);font-family:'DM Mono',monospace;font-size:.7rem;padding:4px 8px;\">folded</span>";

      const stackHTML = playerInfo
        ? `<span style="font-family:'DM Mono',monospace;font-size:.65rem;color:var(--text-muted);">stack: ${fmt(playerInfo.stack)}</span>`
        : "";

      return `<div class="result-row" style="flex-direction:column;align-items:stretch;gap:8px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div class="result-name">${esc(name)}</div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px;">
          <div class="result-delta ${cls}">${sign}${delta.toFixed(1)}</div>
          ${stackHTML}
        </div>
      </div>
      <div style="display:flex;gap:6px;justify-content:center;flex-wrap:wrap;">${cardsHTML}</div>
    </div>`;
    })
    .join("");

  document.getElementById("result-list").innerHTML =
    communityHTML + playersHTML;
  document.getElementById("result-overlay").classList.add("show");

  const parts = entries
    .map(([n, d]) => {
      const cls = d > 0 ? "win" : d < 0 ? "loss" : "";
      const sign = d > 0 ? "+" : "";
      return `${esc(n)}: <span class="${cls}">${sign}${d.toFixed(1)}</span>`;
    })
    .join(" · ");
  addLog(`🃏 Hand ${handNumber} — ${parts}`);
}

function closeResult() {
  document.getElementById("result-overlay").classList.remove("show");
  if (ws) ws.send(JSON.stringify({ type: "hand_ack" }));
}

function showGameOver(stacks) {
  closeResult();
  const sorted = Object.entries(stacks).sort((a, b) => b[1] - a[1]);
  document.getElementById("gameover-stacks").innerHTML = sorted
    .map(
      ([n, s], i) => `
    <div class="result-row" style="margin-bottom:8px;">
      <div class="result-name">${i === 0 ? "🏆 " : ""}${esc(n)}</div>
      <div class="result-delta ${i === 0 ? "pos" : ""}">${fmt(s)} BB</div>
    </div>`,
    )
    .join("");
  document.getElementById("gameover-overlay").classList.add("show");
  addLog(`🏁 <b>Game over!</b> Winner: <b>${sorted[0][0]}</b>`);
}
