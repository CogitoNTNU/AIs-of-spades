/* ── render.js ────────────────────────────────────
   DOM rendering: cards, players, banner, table.
   Depends on: state.js, utils.js
─────────────────────────────────────────────── */

/** Build the HTML for a single playing card. */
function cardHTML(c) {
  const rank = RANKS[c.rank] ?? String(c.rank);
  const suit = SUITS[c.suit] ?? "?";
  const red = c.suit === 2 || c.suit === 1 ? " red" : "";
  return `<div class="card${red}">
    <span class="c-rank">${rank}</span>
    <span class="c-suit">${suit}</span>
    <span class="c-rank-bot">${rank}</span>
  </div>`;
}

/** Render community cards (fills remaining slots with placeholders). */
function renderCommunityCards(cards) {
  const el = document.getElementById("community-cards");
  const shown = cards.map((c) => cardHTML(c)).join("");
  const blanks = Array(5 - cards.length)
    .fill('<div class="card-placeholder">·</div>')
    .join("");
  el.innerHTML = shown + blanks;
}

/** Render the player's own hole cards. */
function renderHandCards(cards) {
  const el = document.getElementById("hand-cards");
  el.innerHTML = cards.length
    ? cards.map((c) => cardHTML(c)).join("")
    : '<div class="card-back">🂠</div><div class="card-back">🂠</div>';
}

/**
 * Render ALL players (including self) sorted by seat order.
 *
 * source = all_players array from server (has name, seat, position, state …)
 * selfObs = observation dict when it's our turn (has player_stack, etc.)
 *
 * Acting order is shown as a small numbered badge on each card.
 * Position badges: SB / BB / UTG / BTN etc.
 */
function renderOthers(allPlayers, legacyOthers) {
  const grid = document.getElementById("players-grid");
  const source = allPlayers || legacyOthers || [];

  if (!source.length) {
    grid.innerHTML =
      "<div style=\"color:var(--text-muted);font-family:'DM Mono',monospace;font-size:.75rem;\">Waiting for players…</div>";
    return;
  }

  const nPlayers = source.length;

  // Sort by seat so the grid is always seat-0, 1, 2 …
  const sorted = [...source].sort((a, b) => (a.seat ?? 0) - (b.seat ?? 0));

  // Build acting order: starting from the player with position 2 (UTG) preflop,
  // or position 0 (SB) post-flop. We just sort by position for the order badge.
  const posOrder = [...sorted].sort(
    (a, b) => (a.position ?? 0) - (b.position ?? 0),
  );
  const actingOrder = {}; // seat → 1-based turn index
  posOrder.forEach((p, i) => {
    actingOrder[p.seat] = i + 1;
  });

  const stateMap = {
    0: ["folded", "badge-folded"],
    1: ["active", "badge-active"],
    2: ["out", "badge-out"],
  };

  const posColors = {
    SB: "#c8a84b", // gold
    BB: "#50a0e8", // blue
    BTN: "#3ab860", // green
    UTG: "#e05050", // red
  };

  grid.innerHTML = sorted
    .map((o) => {
      const isMe = o.seat === mySeat;

      let [stateLabel, badge] = stateMap[o.state] ?? ["?", "badge-out"];
      if (o.is_all_in) {
        stateLabel = "all-in";
        badge = "badge-allin";
      }

      const classes = ["player-card"];
      if (o.state === 0) classes.push("folded");
      if (o.state === 2) classes.push("out");
      if (isMe) classes.push("self");

      const displayName = o.name ? esc(o.name) : `Seat ${(o.seat ?? 0) + 1}`;
      const posLabel = positionLabel(o.position ?? 0, nPlayers);
      const posColor = posColors[posLabel] || "var(--text-muted)";
      const turnNum = actingOrder[o.seat] ?? "";

      return `<div class="${classes.join(" ")}" data-seat="${o.seat}">
      <div class="pc-header">
        <div class="pc-left">
          <div class="pc-turn-badge">${turnNum}</div>
          <div class="pc-name">${displayName}${isMe ? ' <span class="you-tag">YOU</span>' : ""}</div>
        </div>
        <div class="pc-badges">
          <span class="pos-badge" style="color:${posColor};border-color:${posColor};">${posLabel}</span>
          <span class="badge ${badge}">${stateLabel}</span>
        </div>
      </div>
      <div class="pc-stats">
        <div><div class="pc-stat-label">Stack</div>      <div class="pc-stat-value">${fmt(o.stack)}</div></div>
        <div><div class="pc-stat-label">In Pot</div>     <div class="pc-stat-value">${fmt(o.money_in_pot)}</div></div>
        <div><div class="pc-stat-label">Street Bet</div> <div class="pc-stat-value">${fmt(o.bet_this_street)}</div></div>
        <div><div class="pc-stat-label">Turn</div>       <div class="pc-stat-value">#${turnNum}</div></div>
      </div>
    </div>`;
    })
    .join("");
}

/** Update the turn banner and highlight the acting player card. */
function renderTurnBanner(seat, name) {
  const banner = document.getElementById("turn-banner");
  if (seat === mySeat) {
    banner.textContent = "⬡  YOUR TURN";
    banner.classList.add("your-turn");
  } else {
    banner.textContent = `Waiting for ${name}…`;
    banner.classList.remove("your-turn");
  }
  document
    .querySelectorAll(".player-card")
    .forEach((c) => c.classList.remove("acting"));
  const card = document.querySelector(`.player-card[data-seat="${seat}"]`);
  if (card) card.classList.add("acting");
}

/** Render the lobby seat grid. */
function renderSeatsGrid(seated, needed) {
  const grid = document.getElementById("seats-grid");
  grid.innerHTML =
    seated
      .map(
        (name) => `
      <div class="seat-row taken">
        <div class="seat-dot"></div>
        <div class="seat-name">${esc(name)}</div>
        ${name === myName ? '<div class="seat-you">YOU</div>' : ""}
      </div>`,
      )
      .join("") +
    Array(needed)
      .fill(
        `
      <div class="seat-row">
        <div class="seat-dot"></div>
        <div class="seat-name" style="color:var(--text-muted);font-style:italic;">Empty…</div>
      </div>`,
      )
      .join("");
}

/** Apply a table_update message (community cards, pot, others). */
function applyTableUpdate(msg) {
  const streetCards = { 0: 0, 1: 3, 2: 4, 3: 5 };
  const nCards = streetCards[msg.street] ?? 0;

  show("screen-game");
  document.getElementById("pot-value").textContent = fmt(msg.pot);
  document.getElementById("to-call-value").textContent = fmt(msg.bet_to_match);
  document.getElementById("street-name").textContent =
    STREETS[msg.street] ?? "—";

  renderCommunityCards((msg.table_cards || []).slice(0, nCards));

  const source = _mergeMyPlayer(msg.all_players || null);
  renderOthers(source, msg.others || []);
}

/**
 * Merge the local player's own state into an all_players array.
 * This ensures the self card is always rendered with up-to-date data.
 */
function _mergeMyPlayer(allPlayers) {
  if (!allPlayers) return null;
  if (!myPlayerCache) return allPlayers;
  return allPlayers.map((p) =>
    p.seat === mySeat ? { ...p, ...myPlayerCache } : p,
  );
}
