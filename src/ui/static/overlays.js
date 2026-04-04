/* ── overlays.js ──────────────────────────────────
   Hand-result and game-over overlay logic.
   Depends on: state.js, utils.js
─────────────────────────────────────────────── */

function showHandResult(rewards, showdown, players, tableCards) {
  const entries = Object.entries(rewards).sort((a, b) => b[1] - a[1]);

  document.getElementById("result-street").textContent =
    `Hand ${handNumber} of ${totalHands}`;

  let html = "";

  // ── Board ──
  if (tableCards && tableCards.length) {
    html += `
      <div style="margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--felt-rim);">
        <div style="font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:4px;
                    color:var(--text-muted);text-transform:uppercase;margin-bottom:10px;">
          Board
        </div>
        <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
          ${tableCards.map((c) => cardHTML(c)).join("")}
        </div>
      </div>`;
  }

  // ── Players ──
  html += entries
    .map(([name, delta]) => {
      const cls  = delta > 0 ? "pos" : delta < 0 ? "neg" : "zero";
      const sign = delta > 0 ? "+" : "";
      const cards = showdown && showdown[name];
      const playerInfo = players && players.find((p) => p.name === name);

      const cardsHTML = cards
        ? `<div style="display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin-top:8px;">
             ${cards.map((c) => cardHTML(c)).join("")}
           </div>`
        : `<div style="margin-top:6px;font-family:'DM Mono',monospace;font-size:.65rem;
                       color:var(--text-muted);text-align:center;">folded</div>`;

      const stackHTML = playerInfo
        ? `<div style="font-family:'DM Mono',monospace;font-size:.62rem;color:var(--text-muted);
                       margin-top:2px;">stack: ${fmt(playerInfo.stack)}</div>`
        : "";

      // fmt() converts BB → $ — use it for both delta and stack
      const deltaStr = `${sign}${fmt(Math.abs(delta))}`;

      return `
      <div class="result-row" style="flex-direction:column;align-items:stretch;gap:4px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
          <div>
            <div class="result-name">${esc(name)}</div>
            ${stackHTML}
          </div>
          <div class="result-delta ${cls}">${deltaStr}</div>
        </div>
        ${cardsHTML}
      </div>`;
    })
    .join("");

  document.getElementById("result-list").innerHTML = html;
  document.getElementById("result-overlay").classList.add("show");

  const parts = entries
    .map(([n, d]) => {
      const cls  = d > 0 ? "win" : d < 0 ? "loss" : "";
      const sign = d > 0 ? "+" : "";
      return `${esc(n)}: <span class="${cls}">${sign}${fmt(Math.abs(d))}</span>`;
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
      <div class="result-delta ${i === 0 ? "pos" : ""}">${fmt(s)}</div>
    </div>`,
    )
    .join("");
  document.getElementById("gameover-overlay").classList.add("show");
  addLog(`🏁 <b>Game over!</b> Winner: <b>${sorted[0][0]}</b> with ${fmt(sorted[0][1])}`);
}

function showTableReset(stacks, reason) {
  closeResult();
  const sorted = Object.entries(stacks).sort((a, b) => b[1] - a[1]);
  const reasonText =
    reason === "elimination" ? "One player remaining" : "Hands completed";
  document.getElementById("tablereset-reason").textContent = reasonText;
  document.getElementById("tablereset-stacks").innerHTML = sorted
    .map(
      ([n, s], i) => `
    <div class="result-row" style="margin-bottom:8px;">
      <div class="result-name">${i === 0 ? "🏆 " : ""}${esc(n)}</div>
      <div class="result-delta ${i === 0 ? "pos" : ""}">${fmt(s)}</div>
    </div>`,
    )
    .join("");
  document.getElementById("tablereset-overlay").classList.add("show");
  addLog(`🔄 <b>Round over</b> (${reasonText}) — resetting table…`);
}

function closeTableReset() {
  document.getElementById("tablereset-overlay").classList.remove("show");
  handNumber = 0;
  document.getElementById("tb-hand").textContent = `0 / ${totalHands}`;
  if (ws) ws.send(JSON.stringify({ type: "hand_ack" }));
}