# Interactive UI — Start Guide

Launch the browser-based poker table with:

```bash
uv run python src/start_ui_host.py [options]
```

Then open **http://\<your-ip\>:8080** in any browser on the same network.

---

## Scenarios

### 1. Human vs Human

All seats are open for human players. Everyone connects from their own browser and enters a name.

```bash
# 2 players
uv run python src/start_ui_host.py --seats 2

# 4 players, 50 hands, stacks between 100 and 300 BB
uv run python src/start_ui_host.py --seats 4 --hands 50 --stack-low 100 --stack-high 300
```

- The server waits in the lobby until all `--seats` are filled.
- Once the last player joins the game starts automatically.
- After all hands (or one player runs out of chips) a **ROUND OVER** screen appears. Clicking **NEXT ROUND** resets the table with fresh random stacks and loops again.
- Pass `--no-auto-reset` to go back to the lobby instead of looping.

---

### 2. Human vs UGO (AI opponent)

UGO takes one seat; the rest are for humans. Point `--ugo` at a weights file and `--ugo-model` at the matching architecture.

```bash
# EvenNet weights  (default model, no --ugo-model needed)
uv run python src/start_ui_host.py --seats 2 --ugo ui_weights/even_net/weights.pt

# SimoNet weights
uv run python src/start_ui_host.py --seats 3 \
    --ugo ui_weights/simo_net/epoch_10200.pt \
    --ugo-model SimoNet
```

- UGO always occupies **seat 0**; human players fill the remaining seats.
- UGO plays instantly (no delay). Use `--action-delay` to slow it down if needed.
- The UI shows UGO's actions in the Hand History log like any other player.

---

### 3. Watch AI agents play (AI vs AI)

Place model weights under `ui_weights/opponents/` and the server fills every seat with an AI. No human input needed — just spectate.

**Directory layout:**

```
ui_weights/opponents/
└── simo_net/          ← folder name = model class in snake_case
    ├── epoch_2100.pt
    ├── epoch_3500.pt
    └── epoch_5900.pt
```

Each `.pt` file inside a model folder becomes a separate AI player. The folder name is converted to CamelCase to find the model class (`simo_net` → `SimoNet`, `even_net` → `EvenNet`).

```bash
# Explicit opponents dir, 5s action delay (default when --opponents-dir is set)
uv run python src/start_ui_host.py --opponents-dir ui_weights/opponents

# Custom dir and delay
uv run python src/start_ui_host.py \
    --opponents-dir ui_weights/opponents \
    --action-delay 5 \
    --hands 100
```

- Anyone who connects during a running game joins as a **spectator** (read-only view with leaderboard).
- The default action delay is **5 seconds** when using `--opponents-dir`; set `--action-delay 0` to run at full speed.

---

## All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--seats` | `4` | Number of human seats (ignored when `--opponents-dir` is used) |
| `--hands` | `20` | Hands per round before auto-reset |
| `--stack-low` | `50` | Minimum starting stack (big blinds) |
| `--stack-high` | `200` | Maximum starting stack (big blinds) |
| `--ugo` | — | Path to UGO's weights file (enables Human vs UGO mode) |
| `--ugo-model` | `EvenNet` | Model class for UGO (`EvenNet` or `SimoNet`) |
| `--opponents-dir` | — | Directory of AI opponents for AI vs AI mode (must be set explicitly) |
| `--action-delay` | `0` / `5` | Seconds to pause between AI actions (default 5 when using `--opponents-dir`) |
| `--no-auto-reset` | off | Return to lobby after a round instead of looping |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8765` | WebSocket port |
| `--http` | `8080` | HTTP port |
