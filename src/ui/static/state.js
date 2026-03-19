/* ── state.js ─────────────────────────────────────
   Shared mutable state and lookup tables.
   All other modules read/write these variables.
─────────────────────────────────────────────── */
let ws;
let mySeat = null;
let myName = "";
let currentObs = null;
let totalHands = 20;
let handNumber = 0;

// Cache of the last all_players array received from the server.
// Updated on every table_update that includes all_players.
let allPlayersCache = null;

// Cache of the local player's own live state (seat, stack, etc.)
// Updated from player_state and your_turn messages.
let myPlayerCache = null;

// 1 BB = $5
const BB_DOLLARS = 5;

// CardObservation does math.log2(bitmask):
//   clubs=0, diamonds=1, hearts=2, spades=3
const SUITS = { 0: "♣", 1: "♦", 2: "♥", 3: "♠" };
const RANKS = {
  0: "2",
  1: "3",
  2: "4",
  3: "5",
  4: "6",
  5: "7",
  6: "8",
  7: "9",
  8: "10",
  9: "J",
  10: "Q",
  11: "K",
  12: "A",
};
const STREETS = { 0: "Pre-flop", 1: "Flop", 2: "Turn", 3: "River" };

// Position labels — TablePosition enum: SB=0, BB=1, then UTG=2, 3, 4, BTN=n-1
const POSITION_LABELS = {
  0: "SB",
  1: "BB",
  2: "UTG",
  3: "UTG+1",
  4: "UTG+2",
  5: "BTN",
};
function positionLabel(pos, nPlayers) {
  if (pos === 0) return "SB";
  if (pos === 1) return "BB";
  if (pos === nPlayers - 1) return "BTN";
  if (pos === 2) return "UTG";
  return `UTG+${pos - 2}`;
}
