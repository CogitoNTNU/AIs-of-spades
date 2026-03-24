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
      if (!myPlayerCache) myPlayerCache = { seat: mySeat, name: myName };
      if (msg.stack !== undefined) {
        myPlayerCache.stack = msg.stack;
        document.getElementById("my-stack").textContent = fmt(msg.stack);
      }
      break;

    case "table_update":
      if (msg.all_players) {
        allPlayersCache = msg.all_players;
        _updateMyStatsFromAllPlayers(msg.all_players);
      }
      if (!isSpectator) {
        applyTableUpdate(msg);
      }
      if (isSpectator) {
        applySpectatorTableUpdate(msg);
      }
      break;

    case "turn_indicator":
      if (!isSpectator) {
        renderTurnBanner(msg.seat, msg.name);
      }
      if (isSpectator) {
        renderSpectatorTurnBanner(msg.seat, msg.name);
      }
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

    case "spectator_init":
      initSpectator(msg);
      break;

    case "leaderboard_update":
      renderLeaderboard(msg.stacks || {});
      break;
  }
}

/**
 * When a table_update arrives with all_players, find our own entry and
 * update the "Your Position" stats panel + myPlayerCache.
 * This keeps our stack/pot/street-bet current between turns.
 */
function _updateMyStatsFromAllPlayers(allPlayers) {
  if (mySeat === null) return;
  const me = allPlayers.find((p) => p.seat === mySeat);
  if (!me) return;

  if (!myPlayerCache) myPlayerCache = { seat: mySeat, name: myName };
  myPlayerCache.stack = me.stack;
  myPlayerCache.money_in_pot = me.money_in_pot;
  myPlayerCache.bet_this_street = me.bet_this_street;
  myPlayerCache.state = me.state;
  myPlayerCache.position = me.position;

  document.getElementById("my-stack").textContent = fmt(me.stack);
  document.getElementById("my-pot").textContent = fmt(me.money_in_pot);
  document.getElementById("my-street-bet").textContent = fmt(
    me.bet_this_street,
  );
}

function _applyYourTurn(obs) {
  const streetCards = { 0: 0, 1: 3, 2: 4, 3: 5 };
  const nCards = streetCards[obs.street] ?? 0;

  document.getElementById("pot-value").textContent = fmt(obs.pot);
  document.getElementById("to-call-value").textContent = fmt(obs.bet_to_match);
  document.getElementById("street-name").textContent =
    STREETS[obs.street] ?? "—";
  document.getElementById("my-stack").textContent = fmt(obs.player_stack);
  document.getElementById("my-pot").textContent = fmt(obs.player_money_in_pot);
  document.getElementById("my-street-bet").textContent = fmt(
    obs.bet_this_street,
  );

  if (!myPlayerCache) myPlayerCache = { seat: mySeat, name: myName };
  myPlayerCache.stack = obs.player_stack;
  myPlayerCache.money_in_pot = obs.player_money_in_pot;
  myPlayerCache.bet_this_street = obs.bet_this_street;
  myPlayerCache.state = 1;

  renderCommunityCards((obs.table_cards || []).slice(0, nCards));
  renderHandCards(obs.hand_cards || []);

  if (allPlayersCache) {
    const merged = allPlayersCache.map((p) => {
      if (p.seat === mySeat) return { ...p, ...myPlayerCache };
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

  applyActionButtons(obs);
}

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
