/* ── network.js ───────────────────────────────────
   WebSocket connection and top-level message
   dispatcher.
   Depends on: state.js, utils.js, render.js,
               actions.js, overlays.js
─────────────────────────────────────────────── */

function joinTable() {
  const name = document.getElementById("name-input").value.trim();
  if (!name) {
    setJoinError("Please enter a name.");
    return;
  }
  setJoinError("");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.hostname}:{{WS_PORT}}`);
  ws.onopen = () => ws.send(JSON.stringify({ type: "join", name }));
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => addLog("Connection closed.");
  ws.onerror = () => setJoinError("Could not connect to server.");
}

function setJoinError(msg) {
  document.getElementById("join-error").textContent = msg;
}

function handle(msg) {
  switch (msg.type) {
    case "welcome":
      mySeat = msg.seat;
      myName = msg.name;
      totalHands = msg.total_hands ?? 20;
      document.getElementById("tb-name").textContent = myName;
      show("screen-lobby");
      addLog(`Joined as <b>${myName}</b> — seat ${mySeat}`);
      break;

    case "waiting": {
      const seated = msg.seated || [];
      const needed = msg.needed ?? 0;
      const total = seated.length + needed;
      renderSeatsGrid(seated, needed);
      document.getElementById("lobby-subtitle").textContent =
        `${seated.length} of ${total} seat${total !== 1 ? "s" : ""} filled`;
      document.getElementById("lobby-status").textContent =
        needed > 0
          ? `Waiting for ${needed} more player${needed !== 1 ? "s" : ""}…`
          : "Starting…";
      break;
    }

    case "player_state":
      show("screen-game");
      renderHandCards(msg.hand_cards || []);
      // Update self-cache so the player grid shows our current stack
      if (myPlayerCache === null) myPlayerCache = {};
      if (msg.stack !== undefined) {
        myPlayerCache.stack = msg.stack;
        document.getElementById("my-stack").textContent = fmt(msg.stack);
      }
      break;

    case "table_update":
      if (msg.all_players) allPlayersCache = msg.all_players;
      applyTableUpdate(msg);
      break;

    case "turn_indicator":
      renderTurnBanner(msg.seat, msg.name);
      break;

    case "your_turn":
      show("screen-game");
      currentObs = msg.observation;
      _applyYourTurn(msg.observation);
      break;

    case "hand_result":
      handNumber++;
      document.getElementById("tb-hand").textContent =
        `${handNumber} / ${totalHands}`;
      showHandResult(
        msg.rewards,
        msg.showdown || {},
        msg.players || [],
        msg.table_cards || [],
      );
      disableActions();
      break;

    case "hand_log":
      appendHandLog(msg.lines || []);
      break;

    case "game_over":
      showGameOver(msg.final_stacks || {});
      break;

    case "error":
      setJoinError(msg.message);
      addLog(`⚠ ${msg.message}`);
      break;
  }
}

/** Full update when it's this player's turn to act. */
function _applyYourTurn(obs) {
  // Update own stats panel
  document.getElementById("pot-value").textContent = fmt(obs.pot);
  document.getElementById("to-call-value").textContent = fmt(obs.bet_to_match);
  document.getElementById("street-name").textContent =
    STREETS[obs.street] ?? "—";
  document.getElementById("my-stack").textContent = fmt(obs.player_stack);
  document.getElementById("my-pot").textContent = fmt(obs.player_money_in_pot);
  document.getElementById("my-street-bet").textContent = fmt(
    obs.bet_this_street,
  );

  // Keep myPlayerCache in sync so the grid shows our live stats
  if (!myPlayerCache) myPlayerCache = { seat: mySeat, name: myName };
  myPlayerCache.stack = obs.player_stack;
  myPlayerCache.money_in_pot = obs.player_money_in_pot;
  myPlayerCache.bet_this_street = obs.bet_this_street;
  myPlayerCache.state = 1; // ACTIVE (it's our turn)

  renderCommunityCards(obs.table_cards || []);
  renderHandCards(obs.hand_cards || []);

  // Merge live stats from obs.others into allPlayersCache (which has names),
  // then inject our own updated entry so the grid shows all players including self.
  if (allPlayersCache) {
    const merged = allPlayersCache.map((p) => {
      if (p.seat === mySeat) {
        // Use our own live data from the observation
        return { ...p, ...myPlayerCache };
      }
      // obs.others is relative to the acting player — match by position
      const other = obs.others.find((o) => o.position === p.position);
      if (!other) return p;
      return {
        ...p,
        state: other.state,
        stack: other.stack,
        money_in_pot: other.money_in_pot,
        bet_this_street: other.bet_this_street,
        is_all_in: other.is_all_in,
      };
    });
    renderOthers(merged, null);
  } else {
    renderOthers(null, obs.others || []);
  }

  const banner = document.getElementById("turn-banner");
  banner.textContent = "⬡  YOUR TURN";
  banner.classList.add("your-turn");

  // ── Enable action buttons ONLY when it's genuinely our turn ──
  applyActionButtons(obs);

  addLog(
    `<span class="gold">⬡ Your turn</span> — street: <b>${STREETS[obs.street]}</b>, ` +
      `pot: <b>${fmt(obs.pot)}</b>, to call: <b>${fmt(obs.bet_to_match)}</b>`,
  );
}

/**
 * Append real hand-history lines from the server's hh.history to the log panel.
 */
function appendHandLog(lines) {
  lines.forEach((line) => {
    if (!line) return;
    const formatted = esc(line)
      .replace(/(raises|bets|calls|folds|checks)/g, "<b>$1</b>")
      .replace(/\*\*\* ([^*]+) \*\*\*/g, '<span class="gold">*** $1 ***</span>')
      .replace(/collected \$[\d.]+/g, (m) => `<span class="win">${m}</span>`);
    addLog(formatted);
  });
}
