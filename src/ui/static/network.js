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
      // Sent right after reset_hand so every player sees their cards + stack
      // before their turn arrives.
      show("screen-game");
      renderHandCards(msg.hand_cards || []);
      if (msg.stack !== undefined)
        document.getElementById("my-stack").textContent = fmt(msg.stack);
      break;

    case "table_update":
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

    case "game_over":
      showGameOver(msg.final_stacks);
      break;

    case "hand_result":
      handNumber++;
      document.getElementById("tb-hand").textContent =
        `${handNumber} / ${totalHands}`;
      showHandResult(msg.rewards, msg.showdown || {}); // ← aggiungi msg.showdown
      disableActions();
      break;

    case "error":
      setJoinError(msg.message);
      addLog(`⚠ ${msg.message}`);
      break;
  }
}

/** Full update when it's this player's turn to act. */
function _applyYourTurn(obs) {
  document.getElementById("pot-value").textContent = fmt(obs.pot);
  document.getElementById("to-call-value").textContent = fmt(obs.bet_to_match);
  document.getElementById("street-name").textContent =
    STREETS[obs.street] ?? "—";
  document.getElementById("my-stack").textContent = fmt(obs.player_stack);
  document.getElementById("my-pot").textContent = fmt(obs.player_money_in_pot);
  document.getElementById("my-street-bet").textContent = fmt(
    obs.bet_this_street,
  );

  renderCommunityCards(obs.table_cards || []);
  renderHandCards(obs.hand_cards || []);
  renderOthers(obs.others || []);

  const banner = document.getElementById("turn-banner");
  banner.textContent = "⬡  YOUR TURN";
  banner.classList.add("your-turn");

  applyActionButtons(obs);

  addLog(
    `<span class="gold">⬡ Your turn</span> — street: <b>${STREETS[obs.street]}</b>, pot: <b>${fmt(obs.pot)}</b>, to call: <b>${fmt(obs.bet_to_match)}</b>`,
  );
}
