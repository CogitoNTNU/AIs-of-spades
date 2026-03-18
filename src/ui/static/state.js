/* ── state.js ─────────────────────────────────────
   Shared mutable state and lookup tables.
   All other modules read/write these variables.
─────────────────────────────────────────────── */
let ws;
let mySeat    = null;
let myName    = '';
let currentObs = null;
let totalHands = 20;
let handNumber = 0;

// CardObservation does math.log2(bitmask):
//   clubs=0, diamonds=1, hearts=2, spades=3
const SUITS   = { 0:'♣', 1:'♦', 2:'♥', 3:'♠' };
const RANKS   = { 0:'2',1:'3',2:'4',3:'5',4:'6',5:'7',6:'8',7:'9',8:'10',9:'J',10:'Q',11:'K',12:'A' };
const STREETS = { 0:'Pre-flop', 1:'Flop', 2:'Turn', 3:'River' };
