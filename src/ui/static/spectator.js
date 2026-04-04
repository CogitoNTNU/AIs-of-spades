/* ── spectator.js ─────────────────────────────────
   Spectator mode: shown when joining a game in
   progress. Displays the table (community cards,
   pot, players) and a live leaderboard.
   Depends on: state.js, utils.js, render.js
─────────────────────────────────────────────── */

/* ─── Entry point ──────────────────────────────── */

/**
 * Called by network.js when the server sends
 * {"type": "spectator_init"}.
 * Switches to the spectator screen and hydrates
 * the initial state snapshot.
 */
function initSpectator(msg) {
  isSpectator = true; 
  show("screen-spectator");

  if (msg.leaderboard) renderLeaderboard(msg.leaderboard);
  if (msg.table_update) applySpectatorTableUpdate(msg.table_update);
  if (msg.turn_indicator)
    renderSpectatorTurnBanner(msg.turn_indicator.seat, msg.turn_indicator.name);
}

/* ─── Table update (community cards + players) ─── */

function applySpectatorTableUpdate(msg) {
  // Street name
  const streetEl = document.getElementById("sp-street-name");
  if (streetEl) streetEl.textContent = STREETS[msg.street] ?? "—";

  // Pot
  const potEl = document.getElementById("sp-pot-value");
  if (potEl) potEl.textContent = fmt(msg.pot);

  // Community cards
  const ccEl = document.getElementById("sp-community-cards");
  if (ccEl) {
    const n = { 0: 0, 1: 3, 2: 4, 3: 5 }[msg.street] ?? 0;
    const cards = (msg.table_cards || []).slice(0, n);
    ccEl.innerHTML = "";
    for (let i = 0; i < 5; i++) {
      if (i < cards.length) {
        ccEl.appendChild(buildCard(cards[i]));
      } else {
        const ph = document.createElement("div");
        ph.className = "card-placeholder";
        ph.textContent = "·";
        ccEl.appendChild(ph);
      }
    }
  }

  // Players grid (uses all_players if available)
  const players = msg.all_players || null;
  if (players) renderSpectatorPlayers(players, msg.acting_seat ?? null);
}

/* ─── Players grid ─────────────────────────────── */

function renderSpectatorPlayers(players, actingSeat) {
  const grid = document.getElementById("sp-players-grid");
  if (!grid) return;
  grid.innerHTML = "";

  const nPlayers = players.length;
  players.forEach((p) => {
    const isActing = p.seat === actingSeat;
    const stateLabel = _stateLabel(p.state);
    const stateClass = _stateClass(p.state, p.is_all_in);
    const posLabel = _posLabel(p.position, nPlayers);
    const posColor = _posColor(p.position, nPlayers);

    const card = document.createElement("div");
    card.className = `player-card sp-player-card ${stateClass} ${isActing ? "acting" : ""}`;
    card.innerHTML = `
      <div class="pc-header">
        <div class="pc-left">
          <div class="pc-turn-badge">${p.seat + 1}</div>
          <span class="pc-name">${esc(p.name)}</span>
          ${posLabel ? `<span class="pos-badge" style="color:${posColor};border-color:${posColor}44">${posLabel}</span>` : ""}
        </div>
        <div class="pc-badges">
          ${p.is_all_in ? '<span class="badge badge-allin">All-in</span>' : `<span class="badge ${_badgeClass(p.state, p.is_all_in)}">${stateLabel}</span>`}
        </div>
      </div>
      <div class="pc-stats">
        <div>
          <div class="pc-stat-label">Stack</div>
          <div class="pc-stat-value">${fmt(p.stack)}</div>
        </div>
        <div>
          <div class="pc-stat-label">In pot</div>
          <div class="pc-stat-value">${fmt(p.money_in_pot)}</div>
        </div>
        <div>
          <div class="pc-stat-label">Bet</div>
          <div class="pc-stat-value">${fmt(p.bet_this_street)}</div>
        </div>
        <div>
          <div class="pc-stat-label">Action</div>
          <div class="pc-stat-value">${_actionLabel(p.last_action)}</div>
        </div>
        <div>
          <div class="pc-stat-label">Cards</div>
          <div class="pc-stat-value sp-hole-cards" id="sp-cards-${p.seat}"></div>
        </div>
      </div>
    `;
    grid.appendChild(card);

    const cardsEl = card.querySelector(`#sp-cards-${p.seat}`);
    if (p.hand_cards && p.hand_cards.length === 2) {
      p.hand_cards.forEach((c) => cardsEl.appendChild(buildCard(c)));
    } else {
      cardsEl.textContent = "🂠 🂠";
    }
  });
}

/* ─── Turn banner ──────────────────────────────── */

function renderSpectatorTurnBanner(seat, name) {
  const banner = document.getElementById("sp-turn-banner");
  if (!banner) return;
  banner.textContent = `⬡  ${name.toUpperCase()}'S TURN`;
  banner.className = "turn-banner sp-turn-banner";

  // Highlight acting player card
  document.querySelectorAll(".sp-player-card").forEach((el) => {
    el.classList.remove("acting");
  });
  const cards = document.querySelectorAll(".sp-player-card");
  // seat index = card index (rendered in seat order)
  if (cards[seat]) cards[seat].classList.add("acting");
}

/* ─── Leaderboard ──────────────────────────────── */

function renderLeaderboard(stacks) {
  const container = document.getElementById("sp-leaderboard-rows");
  if (!container) return;

  // Sort descending
  const sorted = Object.entries(stacks).sort((a, b) => b[1] - a[1]);
  const maxStack = sorted[0]?.[1] || 1;

  container.innerHTML = sorted
    .map(([name, stack], i) => {
      const pct = Math.round((stack / maxStack) * 100);
      const medal =
        i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : `#${i + 1}`;
      return `
        <div class="lb-row" style="animation-delay:${i * 60}ms">
          <div class="lb-rank">${medal}</div>
          <div class="lb-info">
            <div class="lb-name">${esc(name)}</div>
            <div class="lb-bar-wrap">
              <div class="lb-bar" style="width:${pct}%"></div>
            </div>
          </div>
          <div class="lb-stack">${fmt(stack)}</div>
        </div>
      `;
    })
    .join("");
}

/* ─── Helper: build a single face-down or visible card ─ */

function buildCard(c) {
  const RANKS = [
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "A",
  ];
  const SUITS_SYM = ["♠", "♥", "♦", "♣"];
  const isRed = c.suit === 1 || c.suit === 2;

  const rank = RANKS[c.rank] ?? "?";
  const suit = SUITS_SYM[c.suit] ?? "?";

  const el = document.createElement("div");
  el.className = `card${isRed ? " red" : ""}`;
  el.innerHTML = `
    <span class="c-rank">${rank}</span>
    <span class="c-suit">${suit}</span>
    <span class="c-rank-bot">${rank}</span>
  `;
  return el;
}

/* ─── Action label helper ──────────────────────── */

function _actionLabel(a) {
  if (!a) return "—";
  return a.amount != null ? `${a.label} ${fmt(a.amount)}` : a.label;
}

/* ─── State helpers (shared with render.js logic) ── */

// PlayerState: FOLDED=0, ACTIVE=1, OUT=2, ALL_IN=3
function _stateLabel(state) {
  return state === 0 ? "Folded" : state === 1 ? "Active" : "Out";
}
function _stateClass(state, allIn) {
  if (allIn) return "";
  if (state === 0) return "folded";
  if (state === 2) return "out";
  return "";
}
function _badgeClass(state, allIn) {
  if (allIn) return "badge-allin";
  if (state === 0) return "badge-folded";
  if (state === 2) return "badge-out";
  return "badge-active";
}
// TablePosition: SB=0, BB=1, UTG=2, UTG_1=3, …, BTN=n-1
function _posLabel(pos, nPlayers) {
  if (pos === 0) return "SB";
  if (pos === 1) return "BB";
  if (nPlayers !== undefined && pos === nPlayers - 1) return "BTN";
  if (pos === 2) return "UTG";
  return `UTG+${pos - 2}`;
}
function _posColor(pos, nPlayers) {
  const label = _posLabel(pos, nPlayers);
  const colors = {
    SB: "#c8a84b",
    BB: "#50a0e8",
    BTN: "#3ab860",
    UTG: "#e05050",
    "UTG+1": "#e07030",
    "UTG+2": "#b860e0",
  };
  return colors[label] ?? "#aaa";
}
