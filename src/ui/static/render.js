/* ── render.js ────────────────────────────────────
   DOM rendering: cards, players, banner, table.
   Depends on: state.js, utils.js
─────────────────────────────────────────────── */

/** Build the HTML for a single playing card. */
function cardHTML(c) {
  const rank = RANKS[c.rank] ?? String(c.rank);
  const suit = SUITS[c.suit] ?? '?';
  // hearts=2, diamonds=1 after log2 conversion from treys bitmask
  const red  = (c.suit === 2 || c.suit === 1) ? ' red' : '';
  return `<div class="card${red}">
    <span class="c-rank">${rank}</span>
    <span class="c-suit">${suit}</span>
    <span class="c-rank-bot">${rank}</span>
  </div>`;
}

/** Render community cards (fills remaining slots with placeholders). */
function renderCommunityCards(cards) {
  const el     = document.getElementById('community-cards');
  const shown  = cards.map(c => cardHTML(c)).join('');
  const blanks = Array(5 - cards.length).fill('<div class="card-placeholder">·</div>').join('');
  el.innerHTML = shown + blanks;
}

/** Render the player's own hole cards. */
function renderHandCards(cards) {
  const el = document.getElementById('hand-cards');
  el.innerHTML = cards.length
    ? cards.map(c => cardHTML(c)).join('')
    : '<div class="card-back">🂠</div><div class="card-back">🂠</div>';
}

/** Render the "other players" grid. */
function renderOthers(others) {
  const grid = document.getElementById('players-grid');
  if (!others.length) {
    grid.innerHTML = '<div style="color:var(--text-muted);font-family:\'DM Mono\',monospace;font-size:.75rem;">No other players</div>';
    return;
  }
  const stateMap = {
    0: ['folded', 'badge-folded'],
    1: ['active', 'badge-active'],
    2: ['out',    'badge-out'],
  };
  grid.innerHTML = others.map((o, i) => {
    let [label, badge] = stateMap[o.state] ?? ['?', 'badge-out'];
    if (o.is_all_in) { label = 'all-in'; badge = 'badge-allin'; }

    const classes = ['player-card'];
    if (o.state === 0) classes.push('folded');
    if (o.state === 2) classes.push('out');

    return `<div class="${classes.join(' ')}" data-seat="${i}">
      <div class="pc-header">
        <div class="pc-name">Seat ${i + 1}</div>
        <span class="badge ${badge}">${label}</span>
      </div>
      <div class="pc-stats">
        <div><div class="pc-stat-label">Stack</div>      <div class="pc-stat-value">${fmt(o.stack)}</div></div>
        <div><div class="pc-stat-label">In Pot</div>     <div class="pc-stat-value">${fmt(o.money_in_pot)}</div></div>
        <div><div class="pc-stat-label">Street Bet</div> <div class="pc-stat-value">${fmt(o.bet_this_street)}</div></div>
        <div><div class="pc-stat-label">Position</div>   <div class="pc-stat-value">${o.position}</div></div>
      </div>
    </div>`;
  }).join('');
}

/** Update the turn banner and highlight the acting player card. */
function renderTurnBanner(seat, name) {
  const banner = document.getElementById('turn-banner');
  if (seat === mySeat) {
    banner.textContent = '⬡  YOUR TURN';
    banner.classList.add('your-turn');
  } else {
    banner.textContent = `Waiting for ${name}…`;
    banner.classList.remove('your-turn');
  }
  document.querySelectorAll('.player-card').forEach(c => c.classList.remove('acting'));
  const card = document.querySelector(`.player-card[data-seat="${seat}"]`);
  if (card) card.classList.add('acting');
}

/** Render the lobby seat grid. */
function renderSeatsGrid(seated, needed) {
  const grid = document.getElementById('seats-grid');
  grid.innerHTML =
    seated.map(name => `
      <div class="seat-row taken">
        <div class="seat-dot"></div>
        <div class="seat-name">${esc(name)}</div>
        ${name === myName ? '<div class="seat-you">YOU</div>' : ''}
      </div>`).join('') +
    Array(needed).fill(`
      <div class="seat-row">
        <div class="seat-dot"></div>
        <div class="seat-name" style="color:var(--text-muted);font-style:italic;">Empty…</div>
      </div>`).join('');
}

/** Apply a table_update message (community cards, pot, others). */
function applyTableUpdate(msg) {
  show('screen-game');
  document.getElementById('pot-value').textContent     = fmt(msg.pot);
  document.getElementById('to-call-value').textContent = fmt(msg.bet_to_match);
  document.getElementById('street-name').textContent   = STREETS[msg.street] ?? '—';
  renderCommunityCards(msg.table_cards || []);
  renderOthers(msg.others || []);
}
