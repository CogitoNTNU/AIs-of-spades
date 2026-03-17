_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>POKER LAN</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
/* ═══════════════════════════════════════════════════
   TOKENS
═══════════════════════════════════════════════════ */
:root {
  --felt:        #0a1a0c;
  --felt-mid:    #0e2010;
  --felt-light:  #132815;
  --felt-rim:    #1c3a1e;
  --gold:        #c8a84b;
  --gold-bright: #e8c96a;
  --gold-dim:    #7a6228;
  --cream:       #f0e8d0;
  --cream-dim:   #b8a880;
  --red:         #b83030;
  --red-bright:  #e05050;
  --blue:        #3070b8;
  --blue-bright: #50a0e8;
  --green:       #2a8a48;
  --green-bright:#3ab860;
  --text-muted:  #5a7a5c;
  --chip-w:      #f5f5f0;
  --chip-r:      #cc3333;
  --chip-b:      #2255cc;
  --chip-g:      #228844;
  --chip-blk:    #1a1a1a;

  --r-sm: 6px;
  --r-md: 10px;
  --r-lg: 16px;
  --shadow-card: 2px 4px 12px #00000099, 0 0 0 1px #ffffff08;
  --shadow-panel: 0 8px 32px #00000066;
  --glow-gold: 0 0 20px #c8a84b44;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { font-size: 15px; }

body {
  font-family: 'DM Sans', sans-serif;
  background: var(--felt);
  color: var(--cream);
  min-height: 100vh;
  overflow-x: hidden;

  /* subtle felt texture */
  background-image:
    radial-gradient(ellipse at 50% 0%, #1a3820 0%, transparent 60%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='4' height='4' fill='%230a1a0c'/%3E%3Crect x='0' y='0' width='1' height='1' fill='%230c1e0e' opacity='.6'/%3E%3Crect x='2' y='2' width='1' height='1' fill='%230c1e0e' opacity='.6'/%3E%3C/svg%3E");
}

/* ═══════════════════════════════════════════════════
   LAYOUT SHELLS
═══════════════════════════════════════════════════ */
#screen-join,
#screen-lobby,
#screen-game { display: none; }
#screen-join { display: flex; }

.screen {
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px 16px;
}

/* ═══════════════════════════════════════════════════
   HEADER / LOGO
═══════════════════════════════════════════════════ */
.logo {
  text-align: center;
  margin-bottom: 36px;
}
.logo-suits {
  font-size: 1.6rem;
  letter-spacing: 12px;
  color: var(--gold-dim);
  margin-bottom: 6px;
}
.logo-title {
  font-family: 'Playfair Display', serif;
  font-size: 3.2rem;
  font-weight: 900;
  letter-spacing: 10px;
  color: var(--gold);
  text-shadow: 0 0 40px #c8a84b55, 0 2px 0 #5a3800;
  line-height: 1;
}
.logo-sub {
  font-family: 'DM Mono', monospace;
  font-size: .62rem;
  letter-spacing: 5px;
  color: var(--text-muted);
  margin-top: 8px;
  text-transform: uppercase;
}

/* ═══════════════════════════════════════════════════
   PANEL (generic container)
═══════════════════════════════════════════════════ */
.panel {
  background: var(--felt-mid);
  border: 1px solid var(--felt-rim);
  border-radius: var(--r-lg);
  box-shadow: var(--shadow-panel);
}
.panel-title {
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  letter-spacing: 4px;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 14px;
}

/* ═══════════════════════════════════════════════════
   JOIN FORM
═══════════════════════════════════════════════════ */
.join-card {
  width: 100%;
  max-width: 400px;
  padding: 40px 36px;
}
.join-card h2 {
  font-family: 'Playfair Display', serif;
  font-size: 1.3rem;
  color: var(--gold);
  letter-spacing: 4px;
  margin-bottom: 28px;
  text-align: center;
}

.field-wrap { margin-bottom: 14px; }
.field-label {
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  letter-spacing: 3px;
  color: var(--text-muted);
  text-transform: uppercase;
  display: block;
  margin-bottom: 6px;
}
.field-input {
  width: 100%;
  background: var(--felt);
  border: 1px solid var(--felt-rim);
  color: var(--cream);
  padding: 13px 16px;
  border-radius: var(--r-md);
  font-family: 'DM Mono', monospace;
  font-size: .9rem;
  outline: none;
  transition: border-color .2s, box-shadow .2s;
}
.field-input:focus {
  border-color: var(--gold-dim);
  box-shadow: 0 0 0 3px #c8a84b18;
}
.field-input::placeholder { color: var(--text-muted); }

.btn-join {
  width: 100%;
  margin-top: 6px;
  padding: 14px;
  background: linear-gradient(135deg, var(--gold-dim) 0%, var(--gold) 100%);
  color: #1a0e00;
  border: none;
  border-radius: var(--r-md);
  font-family: 'Playfair Display', serif;
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: 4px;
  cursor: pointer;
  transition: opacity .15s, transform .12s;
  text-transform: uppercase;
}
.btn-join:hover { opacity: .92; transform: translateY(-1px); }
.btn-join:active { transform: translateY(0); }

/* ═══════════════════════════════════════════════════
   LOBBY SCREEN
═══════════════════════════════════════════════════ */
.lobby-card {
  width: 100%;
  max-width: 480px;
  padding: 40px 36px;
  text-align: center;
}
.lobby-card h2 {
  font-family: 'Playfair Display', serif;
  font-size: 1.2rem;
  color: var(--gold);
  letter-spacing: 4px;
  margin-bottom: 6px;
}
.lobby-subtitle {
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  color: var(--text-muted);
  letter-spacing: 3px;
  margin-bottom: 32px;
}
.seats-grid {
  display: grid;
  gap: 10px;
  margin-bottom: 28px;
}
.seat-row {
  display: flex;
  align-items: center;
  gap: 12px;
  background: var(--felt);
  border: 1px solid var(--felt-rim);
  border-radius: var(--r-md);
  padding: 12px 16px;
  transition: border-color .2s;
}
.seat-row.taken { border-color: var(--gold-dim); }
.seat-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--felt-rim);
  flex-shrink: 0;
}
.seat-row.taken .seat-dot { background: var(--gold); box-shadow: 0 0 8px var(--gold); }
.seat-name {
  font-family: 'DM Mono', monospace;
  font-size: .82rem;
  color: var(--cream-dim);
  flex: 1;
  text-align: left;
}
.seat-row.taken .seat-name { color: var(--cream); }
.seat-you {
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  color: var(--gold);
  letter-spacing: 2px;
}
.lobby-status {
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  color: var(--gold-dim);
  letter-spacing: 2px;
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:.6} 50%{opacity:1} }

/* ═══════════════════════════════════════════════════
   GAME SCREEN LAYOUT
═══════════════════════════════════════════════════ */
#screen-game {
  justify-content: flex-start;
  max-width: 820px;
  margin: 0 auto;
  padding: 16px 16px 48px;
  gap: 12px;
}

/* Top bar */
.topbar {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 18px;
  background: var(--felt-mid);
  border: 1px solid var(--felt-rim);
  border-radius: var(--r-md);
  flex-wrap: wrap;
  gap: 8px;
}
.topbar-item { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.topbar-label {
  font-family: 'DM Mono', monospace;
  font-size: .55rem;
  letter-spacing: 3px;
  color: var(--text-muted);
  text-transform: uppercase;
}
.topbar-value {
  font-family: 'Playfair Display', serif;
  font-size: 1rem;
  color: var(--gold);
  font-weight: 700;
}
.topbar-logo {
  font-family: 'Playfair Display', serif;
  font-size: 1.1rem;
  color: var(--gold);
  letter-spacing: 6px;
}

/* Turn banner */
#turn-banner {
  width: 100%;
  text-align: center;
  padding: 8px;
  background: var(--felt-light);
  border: 1px solid var(--felt-rim);
  border-radius: var(--r-sm);
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  letter-spacing: 3px;
  color: var(--text-muted);
  min-height: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .3s;
}
#turn-banner.your-turn {
  border-color: var(--gold);
  color: var(--gold);
  box-shadow: var(--glow-gold);
  animation: turnPulse 1.8s ease-in-out infinite;
}
@keyframes turnPulse { 0%,100%{box-shadow:0 0 12px #c8a84b33} 50%{box-shadow:0 0 24px #c8a84b66} }

/* ═══════════════════════════════════════════════════
   TABLE (community + pot)
═══════════════════════════════════════════════════ */
.table-section {
  width: 100%;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.community-panel { flex: 2; min-width: 240px; padding: 18px 20px; }
.pot-panel { flex: 1; min-width: 150px; padding: 18px 20px; display: flex; flex-direction: column; justify-content: space-between; }

.community-cards { display: flex; gap: 8px; flex-wrap: wrap; min-height: 72px; align-items: flex-end; }

.pot-number {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--gold);
  line-height: 1;
}
.pot-unit {
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  letter-spacing: 3px;
  color: var(--text-muted);
  margin-top: 4px;
}
.to-call-wrap { margin-top: 12px; }
.to-call-label {
  font-family: 'DM Mono', monospace;
  font-size: .55rem;
  letter-spacing: 3px;
  color: var(--text-muted);
  text-transform: uppercase;
}
.to-call-value {
  font-family: 'Playfair Display', serif;
  font-size: 1.2rem;
  color: var(--cream);
}

/* ═══════════════════════════════════════════════════
   CARDS
═══════════════════════════════════════════════════ */
.card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  background: #fdf6e3;
  border-radius: var(--r-sm);
  width: 48px; height: 68px;
  padding: 5px 4px;
  box-shadow: var(--shadow-card);
  font-family: 'Playfair Display', serif;
  font-weight: 700;
  font-size: .95rem;
  color: #1a1a1a;
  user-select: none;
  transition: transform .15s, box-shadow .15s;
  position: relative;
  animation: cardDeal .25s ease-out;
}
@keyframes cardDeal {
  from { transform: translateY(-12px) scale(.9); opacity: 0; }
  to   { transform: translateY(0) scale(1); opacity: 1; }
}
.card:hover { transform: translateY(-4px); box-shadow: 3px 6px 18px #00000099, 0 0 0 1px #ffffff12; }
.card.red { color: var(--red); }
.card .c-rank { font-size: .95rem; line-height: 1; align-self: flex-start; }
.card .c-suit { font-size: 1.2rem; line-height: 1; }
.card .c-rank-bot { font-size: .95rem; line-height: 1; align-self: flex-end; transform: rotate(180deg); }

.card-back {
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #1a3a6a, #0e2040);
  border-radius: var(--r-sm);
  width: 48px; height: 68px;
  box-shadow: var(--shadow-card);
  font-size: 1.4rem;
  border: 2px solid #2a5090;
}
.card-placeholder {
  display: flex; align-items: center; justify-content: center;
  width: 48px; height: 68px;
  border: 1.5px dashed var(--felt-rim);
  border-radius: var(--r-sm);
  color: var(--text-muted);
  font-size: 1.2rem;
}

/* ═══════════════════════════════════════════════════
   YOUR HAND
═══════════════════════════════════════════════════ */
.hand-section {
  width: 100%;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.hand-panel { flex: 1; min-width: 180px; padding: 18px 20px; }
.hand-cards { display: flex; gap: 10px; }

.stack-panel { flex: 2; min-width: 240px; padding: 18px 20px; }
.stack-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.stack-stat { }
.ss-label {
  font-family: 'DM Mono', monospace;
  font-size: .55rem;
  letter-spacing: 3px;
  color: var(--text-muted);
  text-transform: uppercase;
}
.ss-value {
  font-family: 'Playfair Display', serif;
  font-size: 1.4rem;
  color: var(--cream);
  font-weight: 700;
}

/* ═══════════════════════════════════════════════════
   CHIP DISPLAY
═══════════════════════════════════════════════════ */
.chips-row { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px; }
.chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px; height: 28px;
  border-radius: 50%;
  font-family: 'DM Mono', monospace;
  font-size: .55rem;
  font-weight: 500;
  border: 2px dashed rgba(255,255,255,.25);
  box-shadow: 0 2px 4px #00000066;
  line-height: 1;
  flex-shrink: 0;
}
.chip-w { background: var(--chip-w); color: #333; }
.chip-r { background: var(--chip-r); color: #fff; }
.chip-b { background: var(--chip-b); color: #fff; }
.chip-g { background: var(--chip-g); color: #fff; }
.chip-k { background: var(--chip-blk); color: var(--gold); border-color: var(--gold-dim); }

/* ═══════════════════════════════════════════════════
   OTHERS TABLE
═══════════════════════════════════════════════════ */
.others-panel { width: 100%; padding: 18px 20px; }

.players-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }

.player-card {
  background: var(--felt);
  border: 1px solid var(--felt-rim);
  border-radius: var(--r-md);
  padding: 12px 14px;
  transition: border-color .2s;
}
.player-card.acting { border-color: var(--gold-dim); box-shadow: 0 0 12px #c8a84b22; }
.player-card.folded { opacity: .45; }
.player-card.out    { opacity: .25; }

.pc-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.pc-name {
  font-family: 'DM Sans', sans-serif;
  font-size: .85rem;
  font-weight: 500;
  color: var(--cream);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 110px;
}
.badge {
  font-family: 'DM Mono', monospace;
  font-size: .55rem;
  letter-spacing: 1.5px;
  padding: 3px 8px;
  border-radius: 20px;
  text-transform: uppercase;
  font-weight: 500;
  flex-shrink: 0;
}
.badge-active { background: #0d2e14; color: var(--green-bright); border: 1px solid #1a5a24; }
.badge-folded { background: #2e0d0d; color: var(--red-bright);   border: 1px solid #5a1a1a; }
.badge-allin  { background: #2e1e00; color: #f39c12;             border: 1px solid #5a3a00; }
.badge-out    { background: #1a1a1a; color: #555;                border: 1px solid #333; }

.pc-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.pc-stat-label {
  font-family: 'DM Mono', monospace;
  font-size: .52rem;
  letter-spacing: 2px;
  color: var(--text-muted);
  text-transform: uppercase;
}
.pc-stat-value {
  font-family: 'DM Mono', monospace;
  font-size: .82rem;
  color: var(--cream-dim);
}

/* ═══════════════════════════════════════════════════
   ACTION ROW
═══════════════════════════════════════════════════ */
.action-panel { width: 100%; padding: 20px 22px; transition: border-color .3s, box-shadow .3s; }
.action-panel.your-turn { border-color: var(--gold-dim); box-shadow: var(--glow-gold); }

.action-btns { display: flex; gap: 10px; flex-wrap: wrap; }

.btn-action {
  flex: 1;
  min-width: 100px;
  padding: 14px 10px;
  border-radius: var(--r-md);
  font-family: 'DM Sans', sans-serif;
  font-weight: 500;
  font-size: .88rem;
  letter-spacing: 1px;
  cursor: pointer;
  transition: all .15s;
  border: 1.5px solid;
  background: transparent;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
}
.btn-action:disabled { opacity: .2; cursor: not-allowed; pointer-events: none; }
.btn-action .btn-icon { font-size: 1.2rem; }
.btn-action .btn-label { font-size: .78rem; letter-spacing: 2px; text-transform: uppercase; font-family: 'DM Mono', monospace; }
.btn-action .btn-sub { font-size: .65rem; color: inherit; opacity: .7; font-family: 'DM Mono', monospace; }

.btn-check { border-color: var(--green); color: var(--green-bright); }
.btn-check:not(:disabled):hover { background: #2a8a4820; border-color: var(--green-bright); }
.btn-call  { border-color: var(--blue);  color: var(--blue-bright); }
.btn-call:not(:disabled):hover  { background: #3070b820; border-color: var(--blue-bright); }
.btn-fold  { border-color: var(--red);   color: var(--red-bright); }
.btn-fold:not(:disabled):hover  { background: #b8303020; border-color: var(--red-bright); }
.btn-raise { border-color: var(--gold-dim); color: var(--gold); }
.btn-raise:not(:disabled):hover { background: #c8a84b18; border-color: var(--gold); }

/* Bet slider */
.bet-slider-wrap {
  display: none;
  margin-top: 16px;
  background: var(--felt);
  border: 1px solid var(--felt-rim);
  border-radius: var(--r-md);
  padding: 16px 18px;
}
.bet-slider-wrap.open { display: block; }

.bet-amount-display {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--gold);
  text-align: center;
  margin-bottom: 12px;
}
.bet-amount-display span {
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  letter-spacing: 3px;
  color: var(--text-muted);
  display: block;
  margin-bottom: 2px;
}

input[type=range].bet-slider {
  -webkit-appearance: none;
  width: 100%;
  height: 4px;
  background: var(--felt-rim);
  border-radius: 2px;
  outline: none;
  margin: 8px 0;
}
input[type=range].bet-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 20px; height: 20px;
  border-radius: 50%;
  background: var(--gold);
  cursor: pointer;
  box-shadow: 0 0 8px #c8a84b66;
}
input[type=range].bet-slider::-moz-range-thumb {
  width: 20px; height: 20px;
  border-radius: 50%;
  background: var(--gold);
  cursor: pointer;
  border: none;
  box-shadow: 0 0 8px #c8a84b66;
}

.bet-presets { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
.btn-preset {
  background: transparent;
  border: 1px solid var(--felt-rim);
  color: var(--cream-dim);
  padding: 6px 12px;
  border-radius: var(--r-sm);
  font-family: 'DM Mono', monospace;
  font-size: .68rem;
  letter-spacing: 1px;
  cursor: pointer;
  transition: all .15s;
}
.btn-preset:hover { border-color: var(--gold-dim); color: var(--gold); background: #c8a84b10; }

.btn-confirm {
  margin-top: 14px;
  width: 100%;
  padding: 13px;
  background: linear-gradient(135deg, var(--gold-dim), var(--gold));
  color: #1a0e00;
  border: none;
  border-radius: var(--r-md);
  font-family: 'DM Sans', sans-serif;
  font-weight: 700;
  font-size: .9rem;
  letter-spacing: 3px;
  cursor: pointer;
  text-transform: uppercase;
  transition: opacity .15s, transform .1s;
}
.btn-confirm:hover { opacity: .92; transform: translateY(-1px); }

/* ═══════════════════════════════════════════════════
   HAND LOG
═══════════════════════════════════════════════════ */
.log-panel { width: 100%; padding: 18px 20px; }
#log {
  height: 140px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  scrollbar-width: thin;
  scrollbar-color: var(--felt-rim) transparent;
}
#log::-webkit-scrollbar { width: 4px; }
#log::-webkit-scrollbar-thumb { background: var(--felt-rim); border-radius: 2px; }

.log-entry {
  display: flex;
  gap: 10px;
  align-items: baseline;
  padding: 3px 0;
  border-bottom: 1px solid #ffffff04;
  animation: logIn .2s ease-out;
}
@keyframes logIn { from { opacity: 0; transform: translateX(-6px); } to { opacity: 1; transform: none; } }
.log-time {
  font-family: 'DM Mono', monospace;
  font-size: .58rem;
  color: var(--text-muted);
  flex-shrink: 0;
}
.log-msg {
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  color: var(--cream-dim);
  line-height: 1.4;
}
.log-msg b { color: var(--cream); font-weight: 500; }
.log-msg .win { color: var(--green-bright); }
.log-msg .loss { color: var(--red-bright); }
.log-msg .gold { color: var(--gold); }

/* ═══════════════════════════════════════════════════
   HAND RESULT OVERLAY
═══════════════════════════════════════════════════ */
#result-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: #00000088;
  backdrop-filter: blur(4px);
  z-index: 100;
  align-items: center;
  justify-content: center;
}
#result-overlay.show { display: flex; }
.result-card {
  background: var(--felt-mid);
  border: 1px solid var(--gold-dim);
  border-radius: var(--r-lg);
  padding: 36px 40px;
  min-width: 320px;
  max-width: 480px;
  text-align: center;
  box-shadow: 0 20px 60px #00000099, var(--glow-gold);
  animation: resultIn .3s cubic-bezier(.34,1.4,.64,1);
}
@keyframes resultIn { from { transform: scale(.85); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.result-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem;
  color: var(--gold);
  letter-spacing: 4px;
  margin-bottom: 6px;
}
.result-subtitle {
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  letter-spacing: 3px;
  color: var(--text-muted);
  margin-bottom: 28px;
}
.result-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 28px; }
.result-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--felt);
  border-radius: var(--r-sm);
  padding: 10px 16px;
}
.result-name { font-family: 'DM Sans', sans-serif; font-size: .88rem; }
.result-delta {
  font-family: 'Playfair Display', serif;
  font-size: 1.1rem;
  font-weight: 700;
}
.result-delta.pos { color: var(--green-bright); }
.result-delta.neg { color: var(--red-bright); }
.result-delta.zero { color: var(--text-muted); }
.btn-continue {
  padding: 12px 32px;
  background: var(--felt-rim);
  border: 1px solid var(--gold-dim);
  color: var(--gold);
  border-radius: var(--r-md);
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  letter-spacing: 3px;
  cursor: pointer;
  text-transform: uppercase;
  transition: background .15s;
}
.btn-continue:hover { background: var(--felt-light); }

/* ═══════════════════════════════════════════════════
   GAME OVER OVERLAY
═══════════════════════════════════════════════════ */
#gameover-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: #00000099;
  backdrop-filter: blur(6px);
  z-index: 200;
  align-items: center;
  justify-content: center;
}
#gameover-overlay.show { display: flex; }
.gameover-card {
  background: var(--felt-mid);
  border: 1px solid var(--gold);
  border-radius: var(--r-lg);
  padding: 48px 44px;
  min-width: 360px;
  max-width: 520px;
  text-align: center;
  box-shadow: 0 24px 80px #00000099, var(--glow-gold);
  animation: resultIn .4s cubic-bezier(.34,1.3,.64,1);
}
.gameover-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.2rem;
  color: var(--gold);
  letter-spacing: 6px;
  margin-bottom: 4px;
}
.gameover-suits { font-size: 1.4rem; letter-spacing: 8px; color: var(--gold-dim); margin-bottom: 28px; }

/* ═══════════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════════ */
@media(max-width:500px) {
  .logo-title { font-size: 2.2rem; letter-spacing: 6px; }
  .action-btns { flex-direction: column; }
  .btn-action { min-width: unset; }
}
</style>
</head>
<body>

<!-- ══ JOIN SCREEN ══ -->
<div id="screen-join" class="screen">
  <div class="logo">
    <div class="logo-suits">♠ ♥ ♦ ♣</div>
    <div class="logo-title">POKER</div>
    <div class="logo-sub">Texas Hold'em · LAN Edition</div>
  </div>
  <div class="panel join-card">
    <h2>TAKE A SEAT</h2>
    <div class="field-wrap">
      <label class="field-label" for="name-input">Your Name</label>
      <input id="name-input" class="field-input" type="text" maxlength="20"
             placeholder="Enter a name…" autocomplete="off"
             onkeydown="if(event.key==='Enter')joinTable()" />
    </div>
    <button class="btn-join" onclick="joinTable()">JOIN TABLE</button>
    <div id="join-error" style="margin-top:12px;font-family:'DM Mono',monospace;font-size:.7rem;color:var(--red-bright);text-align:center;min-height:18px;"></div>
  </div>
</div>

<!-- ══ LOBBY SCREEN ══ -->
<div id="screen-lobby" class="screen">
  <div class="logo" style="margin-bottom:20px;">
    <div class="logo-title" style="font-size:2rem;">POKER</div>
  </div>
  <div class="panel lobby-card">
    <h2>WAITING ROOM</h2>
    <div class="lobby-subtitle" id="lobby-subtitle">Filling seats…</div>
    <div class="seats-grid" id="seats-grid"></div>
    <div class="lobby-status" id="lobby-status">Waiting for players…</div>
  </div>
</div>

<!-- ══ GAME SCREEN ══ -->
<div id="screen-game" class="screen">

  <!-- Top bar -->
  <div class="topbar">
    <div class="topbar-item">
      <div class="topbar-label">Player</div>
      <div class="topbar-value" id="tb-name">—</div>
    </div>
    <div class="topbar-logo">♠ POKER ♠</div>
    <div class="topbar-item">
      <div class="topbar-label">Hand</div>
      <div class="topbar-value" id="tb-hand">— / —</div>
    </div>
  </div>

  <!-- Turn banner -->
  <div id="turn-banner">Waiting for game to start…</div>

  <!-- Community + Pot -->
  <div class="table-section">
    <div class="panel community-panel">
      <div class="panel-title">Community Cards</div>
      <div class="community-cards" id="community-cards">
        <div class="card-placeholder">♠</div>
        <div class="card-placeholder">♥</div>
        <div class="card-placeholder">♦</div>
        <div class="card-placeholder">♣</div>
        <div class="card-placeholder">?</div>
      </div>
    </div>
    <div class="panel pot-panel">
      <div>
        <div class="panel-title">Pot</div>
        <div class="pot-number" id="pot-value">0.0</div>
        <div class="pot-unit">BIG BLINDS</div>
      </div>
      <div class="to-call-wrap">
        <div class="to-call-label">To Call</div>
        <div class="to-call-value" id="to-call-value">—</div>
      </div>
    </div>
  </div>

  <!-- Your hand + stats -->
  <div class="hand-section">
    <div class="panel hand-panel">
      <div class="panel-title">Your Hand</div>
      <div class="hand-cards" id="hand-cards">
        <div class="card-back">🂠</div>
        <div class="card-back">🂠</div>
      </div>
    </div>
    <div class="panel stack-panel">
      <div class="panel-title">Your Position</div>
      <div class="stack-grid">
        <div class="stack-stat">
          <div class="ss-label">Stack</div>
          <div class="ss-value" id="my-stack">—</div>
        </div>
        <div class="stack-stat">
          <div class="ss-label">In Pot</div>
          <div class="ss-value" id="my-pot">—</div>
        </div>
        <div class="stack-stat">
          <div class="ss-label">Street Bet</div>
          <div class="ss-value" id="my-street-bet">—</div>
        </div>
        <div class="stack-stat">
          <div class="ss-label">Street</div>
          <div class="ss-value" id="street-name">—</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Others -->
  <div class="panel others-panel">
    <div class="panel-title">Other Players</div>
    <div class="players-grid" id="players-grid">
      <div style="color:var(--text-muted);font-family:'DM Mono',monospace;font-size:.75rem;">Waiting for players…</div>
    </div>
  </div>

  <!-- Actions -->
  <div class="panel action-panel" id="action-panel">
    <div class="panel-title">Your Action</div>
    <div class="action-btns">
      <button id="btn-checkcall" class="btn-action btn-check" onclick="doCheckCall()">
        <span class="btn-icon">✓</span>
        <span class="btn-label">CHECK</span>
      </button>
      <button id="btn-fold" class="btn-action btn-fold" onclick="doAction('fold')" disabled>
        <span class="btn-icon">✕</span>
        <span class="btn-label">FOLD</span>
      </button>
      <button id="btn-raise" class="btn-action btn-raise" onclick="toggleBetSlider()" disabled>
        <span class="btn-icon">↑</span>
        <span class="btn-label">BET / RAISE</span>
      </button>
    </div>

    <div class="bet-slider-wrap" id="bet-slider-wrap">
      <div class="bet-amount-display">
        <span>Bet Amount</span>
        <span id="bet-display-val">0.0</span>
      </div>
      <input type="range" class="bet-slider" id="bet-slider"
             min="0" max="100" step="0.5" oninput="syncBetSlider()" />
      <div class="bet-presets" id="bet-presets"></div>
      <button class="btn-confirm" onclick="confirmBet()">CONFIRM BET</button>
    </div>
  </div>

  <!-- Log -->
  <div class="panel log-panel">
    <div class="panel-title">Hand History</div>
    <div id="log"></div>
  </div>

</div>

<!-- Hand result overlay -->
<div id="result-overlay">
  <div class="result-card">
    <div class="result-title">HAND OVER</div>
    <div class="result-subtitle" id="result-street">—</div>
    <div class="result-list" id="result-list"></div>
    <button class="btn-continue" onclick="closeResult()">NEXT HAND</button>
  </div>
</div>

<!-- Game over overlay -->
<div id="gameover-overlay">
  <div class="gameover-card">
    <div class="gameover-title">GAME OVER</div>
    <div class="gameover-suits">♠ ♥ ♦ ♣</div>
    <div id="gameover-stacks" style="margin-bottom:28px;"></div>
    <button class="btn-continue" onclick="location.reload()">NEW GAME</button>
  </div>
</div>

<script>
/* ═══════════════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════════════ */
let ws;
let mySeat = null;
let myName = '';
let currentObs = null;
let totalHands = 20;
let handNumber = 0;
let seatCount = 0;  // filled by welcome msg or seats grid

// CardObservation does math.log2(bitmask): clubs=0, diamonds=1, hearts=2, spades=3
const SUITS  = { 0:'♣', 1:'♦', 2:'♥', 3:'♠' };
const RANKS  = { 0:'2',1:'3',2:'4',3:'5',4:'6',5:'7',6:'8',7:'9',8:'10',9:'J',10:'Q',11:'K',12:'A' };
const STREETS = { 0:'Pre-flop', 1:'Flop', 2:'Turn', 3:'River' };

/* ═══════════════════════════════════════════════════════════════════
   SCREENS
═══════════════════════════════════════════════════════════════════ */
function show(id) {
  ['screen-join','screen-lobby','screen-game'].forEach(s =>
    document.getElementById(s).style.display = 'none');
  const el = document.getElementById(id);
  el.style.display = 'flex';
}

/* ═══════════════════════════════════════════════════════════════════
   JOIN
═══════════════════════════════════════════════════════════════════ */
function joinTable() {
  const name = document.getElementById('name-input').value.trim();
  if (!name) { setJoinError('Please enter a name.'); return; }
  setJoinError('');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.hostname}:{{WS_PORT}}`);
  ws.onopen    = () => ws.send(JSON.stringify({ type: 'join', name }));
  ws.onmessage = e => handle(JSON.parse(e.data));
  ws.onclose   = () => { addLog('Connection closed.'); };
  ws.onerror   = () => setJoinError('Could not connect to server.');
}

function setJoinError(msg) {
  document.getElementById('join-error').textContent = msg;
}

/* ═══════════════════════════════════════════════════════════════════
   MESSAGE HANDLER
═══════════════════════════════════════════════════════════════════ */
function handle(msg) {
  switch (msg.type) {

    case 'welcome':
      mySeat     = msg.seat;
      myName     = msg.name;
      totalHands = msg.total_hands ?? 20;
      document.getElementById('tb-name').textContent = myName;
      show('screen-lobby');
      addLog(`Joined as <b>${myName}</b> — seat ${mySeat}`);
      break;

    case 'waiting': {
      const seated = msg.seated || [];
      const needed = msg.needed ?? 0;
      seatCount = seated.length + needed;
      renderSeatsGrid(seated, needed);
      document.getElementById('lobby-subtitle').textContent =
        `${seated.length} of ${seatCount} seat${seatCount !== 1 ? 's' : ''} filled`;
      document.getElementById('lobby-status').textContent =
        needed > 0 ? `Waiting for ${needed} more player${needed !== 1 ? 's' : ''}…` : 'Starting…';
      break;
    }

    case 'table_update':
      applyTableUpdate(msg);
      break;

    case 'turn_indicator':
      renderTurnBanner(msg.seat, msg.name);
      break;

    case 'your_turn':
      show('screen-game');
      currentObs = msg.observation;
      applyYourTurn(msg.observation);
      break;

    case 'hand_result':
      handNumber++;
      document.getElementById('tb-hand').textContent = `${handNumber} / ${totalHands}`;
      showHandResult(msg.rewards);
      disableActions();
      break;

    case 'game_over':
      showGameOver(msg.final_stacks);
      break;

    case 'error':
      setJoinError(msg.message);
      addLog(`⚠ ${msg.message}`);
      break;
  }
}

/* ═══════════════════════════════════════════════════════════════════
   LOBBY
═══════════════════════════════════════════════════════════════════ */
function renderSeatsGrid(seated, needed) {
  const grid = document.getElementById('seats-grid');
  grid.innerHTML = '';
  seated.forEach((name, i) => {
    const isMe = name === myName;
    grid.innerHTML += `
      <div class="seat-row taken">
        <div class="seat-dot"></div>
        <div class="seat-name">${esc(name)}</div>
        ${isMe ? '<div class="seat-you">YOU</div>' : ''}
      </div>`;
  });
  for (let i = 0; i < needed; i++) {
    grid.innerHTML += `
      <div class="seat-row">
        <div class="seat-dot"></div>
        <div class="seat-name" style="color:var(--text-muted);font-style:italic;">Empty…</div>
      </div>`;
  }
}

/* ═══════════════════════════════════════════════════════════════════
   TABLE UPDATE  (community cards, pot, others — no action)
═══════════════════════════════════════════════════════════════════ */
function applyTableUpdate(msg) {
  show('screen-game');
  document.getElementById('pot-value').textContent     = fmt(msg.pot);
  document.getElementById('to-call-value').textContent = fmt(msg.bet_to_match);
  document.getElementById('street-name').textContent   = STREETS[msg.street] ?? '—';
  renderCommunityCards(msg.table_cards || []);
  renderOthers(msg.others || [], null);
}

/* ═══════════════════════════════════════════════════════════════════
   YOUR TURN
═══════════════════════════════════════════════════════════════════ */
function applyYourTurn(obs) {
  // Pot / bet
  document.getElementById('pot-value').textContent      = fmt(obs.pot);
  document.getElementById('to-call-value').textContent  = fmt(obs.bet_to_match);
  document.getElementById('street-name').textContent    = STREETS[obs.street] ?? '—';
  document.getElementById('my-stack').textContent       = fmt(obs.player_stack);
  document.getElementById('my-pot').textContent         = fmt(obs.player_money_in_pot);
  document.getElementById('my-street-bet').textContent  = fmt(obs.bet_this_street);

  renderCommunityCards(obs.table_cards || []);
  renderHandCards(obs.hand_cards || []);
  renderOthers(obs.others || [], mySeat);

  // Turn banner
  const banner = document.getElementById('turn-banner');
  banner.textContent = '⬡  YOUR TURN';
  banner.classList.add('your-turn');

  // Action buttons
  const a = obs.actions;
  const cc = document.getElementById('btn-checkcall');
  document.getElementById('action-panel').classList.add('your-turn');

  if (a.check) {
    cc.disabled = false;
    cc.className = 'btn-action btn-check';
    cc.innerHTML = `<span class="btn-icon">✓</span><span class="btn-label">CHECK</span>`;
    cc.dataset.mode = 'check';
  } else if (a.call) {
    cc.disabled = false;
    cc.className = 'btn-action btn-call';
    cc.innerHTML = `<span class="btn-icon">↩</span><span class="btn-label">CALL</span><span class="btn-sub">${fmt(obs.bet_to_match)} BB</span>`;
    cc.dataset.mode = 'call';
  } else {
    cc.disabled = true;
    cc.className = 'btn-action btn-check';
    cc.innerHTML = `<span class="btn-icon">✓</span><span class="btn-label">CHECK</span>`;
    cc.dataset.mode = 'check';
  }

  document.getElementById('btn-fold').disabled  = !a.fold;
  document.getElementById('btn-raise').disabled = !a.bet;

  // Slider
  const sl = document.getElementById('bet-slider');
  sl.min   = obs.bet_range.lower;
  sl.max   = obs.bet_range.upper;
  sl.value = obs.bet_range.lower;
  document.getElementById('bet-display-val').textContent = fmt(obs.bet_range.lower);
  buildPresets(obs.bet_range.lower, obs.bet_range.upper, obs.pot, obs.player_stack);

  // Close slider if open
  document.getElementById('bet-slider-wrap').classList.remove('open');

  addLog(`<span class="gold">⬡ Your turn</span> — street: <b>${STREETS[obs.street]}</b>, pot: <b>${fmt(obs.pot)}</b>, to call: <b>${fmt(obs.bet_to_match)}</b>`);
}

/* ═══════════════════════════════════════════════════════════════════
   TURN INDICATOR  (from turn_indicator message)
═══════════════════════════════════════════════════════════════════ */
function renderTurnBanner(seat, name) {
  const banner = document.getElementById('turn-banner');
  if (seat === mySeat) {
    banner.textContent = '⬡  YOUR TURN';
    banner.classList.add('your-turn');
  } else {
    banner.textContent = `Waiting for ${name}…`;
    banner.classList.remove('your-turn');
  }
  // update others highlighting
  document.querySelectorAll('.player-card').forEach(c => c.classList.remove('acting'));
  const card = document.querySelector(`.player-card[data-seat="${seat}"]`);
  if (card) card.classList.add('acting');
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER HELPERS
═══════════════════════════════════════════════════════════════════ */
function renderCommunityCards(cards) {
  const el = document.getElementById('community-cards');
  if (!cards.length) {
    el.innerHTML = [1,2,3,4,5].map(() => '<div class="card-placeholder">·</div>').join('');
    return;
  }
  const shown = cards.map(c => cardHTML(c)).join('');
  const blanks = [1,2,3,4,5].slice(cards.length).map(() => '<div class="card-placeholder">·</div>').join('');
  el.innerHTML = shown + blanks;
}

function renderHandCards(cards) {
  const el = document.getElementById('hand-cards');
  if (!cards.length) {
    el.innerHTML = '<div class="card-back">🂠</div><div class="card-back">🂠</div>';
    return;
  }
  el.innerHTML = cards.map(c => cardHTML(c)).join('');
}

function renderOthers(others, actingSeat) {
  const grid = document.getElementById('players-grid');
  if (!others.length) {
    grid.innerHTML = '<div style="color:var(--text-muted);font-family:\'DM Mono\',monospace;font-size:.75rem;">No other players</div>';
    return;
  }
  grid.innerHTML = others.map((o, i) => {
    const isAllin = o.is_all_in;
    const stateMap = { 0: ['folded','badge-folded'], 1: ['active','badge-active'], 2: ['out','badge-out'] };
    let [label, badge] = stateMap[o.state] ?? ['?','badge-out'];
    if (isAllin) { label = 'all-in'; badge = 'badge-allin'; }

    const classes = ['player-card'];
    if (o.state === 0) classes.push('folded');
    if (o.state === 2) classes.push('out');

    return `<div class="${classes.join(' ')}" data-seat="${i}">
      <div class="pc-header">
        <div class="pc-name">Seat ${i + 1}</div>
        <span class="badge ${badge}">${label}</span>
      </div>
      <div class="pc-stats">
        <div>
          <div class="pc-stat-label">Stack</div>
          <div class="pc-stat-value">${fmt(o.stack)}</div>
        </div>
        <div>
          <div class="pc-stat-label">In Pot</div>
          <div class="pc-stat-value">${fmt(o.money_in_pot)}</div>
        </div>
        <div>
          <div class="pc-stat-label">Street Bet</div>
          <div class="pc-stat-value">${fmt(o.bet_this_street)}</div>
        </div>
        <div>
          <div class="pc-stat-label">Position</div>
          <div class="pc-stat-value">${o.position}</div>
        </div>
      </div>
    </div>`;
  }).join('');
}

/* ═══════════════════════════════════════════════════════════════════
   CARD HTML
═══════════════════════════════════════════════════════════════════ */
function cardHTML(c) {
  const rank = RANKS[c.rank] ?? String(c.rank);
  const suit = SUITS[c.suit] ?? '?';
  // hearts=2, diamonds=1 after log2 conversion
  const red  = (c.suit === 2 || c.suit === 1) ? ' red' : '';
  return `<div class="card${red}">
    <span class="c-rank">${rank}</span>
    <span class="c-suit">${suit}</span>
    <span class="c-rank-bot">${rank}</span>
  </div>`;
}

/* ═══════════════════════════════════════════════════════════════════
   ACTIONS
═══════════════════════════════════════════════════════════════════ */
function doCheckCall() {
  const mode = document.getElementById('btn-checkcall').dataset.mode || 'check';
  doAction(mode);
}

function doAction(type) {
  if (!ws) return;
  disableActions();
  ws.send(JSON.stringify({ type: 'action', action_type: type, bet_amount: 0 }));
  document.getElementById('turn-banner').classList.remove('your-turn');
  document.getElementById('turn-banner').textContent = 'Waiting…';
  document.getElementById('action-panel').classList.remove('your-turn');
  addLog(`↑ <b>${type.toUpperCase()}</b>`);
}

function toggleBetSlider() {
  document.getElementById('bet-slider-wrap').classList.toggle('open');
}

function syncBetSlider() {
  const val = parseFloat(document.getElementById('bet-slider').value);
  document.getElementById('bet-display-val').textContent = fmt(val);
}

function buildPresets(lower, upper, pot, stack) {
  const presets = [
    { label: '½ pot', val: pot * 0.5 },
    { label: '¾ pot', val: pot * 0.75 },
    { label: '1× pot', val: pot },
    { label: '2× pot', val: pot * 2 },
    { label: 'All-in', val: upper },
  ];
  document.getElementById('bet-presets').innerHTML = presets
    .filter(p => p.val >= lower && p.val <= upper + 0.01)
    .map(p => {
      const v = Math.min(upper, Math.max(lower, Math.round(p.val * 2) / 2));
      return `<button class="btn-preset" onclick="setPreset(${v})">${p.label}<br><small>${fmt(v)}</small></button>`;
    }).join('');
}

function setPreset(val) {
  const sl = document.getElementById('bet-slider');
  val = Math.max(+sl.min, Math.min(+sl.max, val));
  sl.value = val;
  document.getElementById('bet-display-val').textContent = fmt(val);
}

function confirmBet() {
  if (!ws) return;
  const amount = parseFloat(document.getElementById('bet-slider').value);
  disableActions();
  ws.send(JSON.stringify({ type: 'action', action_type: 'bet', bet_amount: amount }));
  document.getElementById('turn-banner').classList.remove('your-turn');
  document.getElementById('turn-banner').textContent = 'Waiting…';
  document.getElementById('action-panel').classList.remove('your-turn');
  addLog(`↑ <b>BET</b> ${fmt(amount)} BB`);
}

function disableActions() {
  ['btn-checkcall','btn-fold','btn-raise'].forEach(id => {
    document.getElementById(id).disabled = true;
  });
  document.getElementById('bet-slider-wrap').classList.remove('open');
}

/* ═══════════════════════════════════════════════════════════════════
   OVERLAYS
═══════════════════════════════════════════════════════════════════ */
function showHandResult(rewards) {
  const overlay = document.getElementById('result-overlay');
  const list    = document.getElementById('result-list');
  const subtitle= document.getElementById('result-street');

  subtitle.textContent = `Hand ${handNumber} of ${totalHands}`;

  // Sort: winners first
  const entries = Object.entries(rewards).sort((a,b) => b[1]-a[1]);
  list.innerHTML = entries.map(([name, delta]) => {
    const cls = delta > 0 ? 'pos' : delta < 0 ? 'neg' : 'zero';
    const sign = delta > 0 ? '+' : '';
    return `<div class="result-row">
      <div class="result-name">${esc(name)}</div>
      <div class="result-delta ${cls}">${sign}${delta.toFixed(1)}</div>
    </div>`;
  }).join('');

  overlay.classList.add('show');

  // Log it
  const parts = entries.map(([n,d]) => {
    const cls = d > 0 ? 'win' : d < 0 ? 'loss' : '';
    const sign = d > 0 ? '+' : '';
    return `${esc(n)}: <span class="${cls}">${sign}${d.toFixed(1)}</span>`;
  }).join(' · ');
  addLog(`🃏 Hand ${handNumber} — ${parts}`);
}

function closeResult() {
  document.getElementById('result-overlay').classList.remove('show');
}

function showGameOver(stacks) {
  closeResult();
  document.getElementById('gameover-overlay').classList.add('show');
  const sorted = Object.entries(stacks).sort((a,b) => b[1]-a[1]);
  document.getElementById('gameover-stacks').innerHTML =
    sorted.map(([n,s], i) => `
      <div class="result-row" style="margin-bottom:8px;">
        <div class="result-name">${i===0?'🏆 ':''}${esc(n)}</div>
        <div class="result-delta ${i===0?'pos':''}">${fmt(s)} BB</div>
      </div>`
    ).join('');
  addLog(`🏁 <b>Game over!</b> Winner: <b>${sorted[0][0]}</b>`);
}

/* ═══════════════════════════════════════════════════════════════════
   LOG
═══════════════════════════════════════════════════════════════════ */
function addLog(msg) {
  const log = document.getElementById('log');
  const ts  = new Date().toLocaleTimeString('it',{ hour:'2-digit', minute:'2-digit', second:'2-digit' });
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<span class="log-time">${ts}</span><span class="log-msg">${msg}</span>`;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

/* ═══════════════════════════════════════════════════════════════════
   UTILS
═══════════════════════════════════════════════════════════════════ */
function fmt(n) {
  if (n === undefined || n === null) return '—';
  const v = Number(n);
  return isNaN(v) ? '—' : v.toFixed(1);
}

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;');
}
</script>
</body>
</html>
"""
