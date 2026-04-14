# MiheerNet

Hybrid reinforcement-learning-ready Poker network that fuses:
- **CardsEncoder (CNN)** for card grid [B,4,4,13]
- **BetsTransformer (Transformer)** for betting history [B,64,8] (or flat [B,512])
- **StateMLP (MLP)** for internal hand/game state
- **Fusion trunk** producing policy logits + bet mean/std and refreshed internal states

## File layout

- `miheer_net.py` — main `MiheerNet` class (inherits `PokerNet`)
- `_cards_encoder.py` — residual CNN encoder with global pooling + projection
- `_bets_transformer.py` — Transformer encoder with sinusoidal positions, CLS, flexible pooling
- `_state_mlp.py` — dual-branch MLP for hand/game state, fused projection
- `preprocess.py` — converts `Observation` to tensors for all branches
- `architecture.txt` — quick ASCII diagram
- `__init__.py` — exports

## Expected inputs

- Cards grid: `[B, 4, 4, 13]` one-hot (batch, slot, suit, rank). Slots: 0=hand, 1-3=community (flop/turn/river mapping).
- Betting history: `[B, 64, 8]` (time, features) or flattened `[B, 512]`. CLS token is prepended internally; `max_seq_len` is set to 65.
- Internal state: `hand_state` `[B, H]`, `game_state` `[B, G]` maintained inside the net and attached to observations for replay.

## Default dimensions

- Cards encoder output: 128
- Bets encoder output: 128 (model_dim)
- State encoder output: 96
- Fusion trunk hidden: 256
- Policy head: 3 logits (fold/call/raise)
- Bet heads: mean in [0,1] (via sigmoid), logvar → std clamped to [1e-4, 1.0]
- Internal state dims: hand_state_dim=32, game_state_dim=32

## Usage example

```python
from nn.miheer_net import MiheerNet
from pokerenv.observation import Observation

net = MiheerNet()
net.initialize_internal_state(batch_size=1)

obs: Observation = ...  # must carry hand_log shaped [64,8] (or flat [512]) and card info
action_logits, bet_mean, bet_std = net(obs)

# on new hand
net.new_hand(batch_size=1)
```

## Preprocessing notes

- `preprocess_observation` handles card encoding, betting log conversion, optional replay states, and batching.
- Betting logs are treated as dense (no padding mask). If you introduce variable-length logs, pass/construct a padding mask and extend the transformer accordingly.

## Suggested config stub (YAML)

```yaml
model: MiheerNet
model_args:
  cards_channels: [32, 64, 96]
  cards_out_dim: 128
  cards_dropout: 0.05
  bet_seq_len: 64
  bet_feature_dim: 8
  bet_model_dim: 128
  bet_heads: 4
  bet_layers: 3
  bet_ff_dim: 256
  bet_dropout: 0.1
  bet_pooling: cls
  hand_state_dim: 32
  game_state_dim: 32
  state_hidden_dim: 128
  state_out_dim: 96
  state_dropout: 0.1
  trunk_hidden_dim: 256
  trunk_dropout: 0.1
```

## Tips for extension

- Change `bet_seq_len`/`bet_feature_dim` if your `hand_log` schema changes; update preprocessing accordingly.
- Switch `bet_pooling` to `mean` or `max` for different aggregation behaviors.
- Enable/disable residuals, norms, or dropout in encoders for ablation studies.
- To widen capacity, increase `cards_channels`, `bet_model_dim`, `state_hidden_dim`, and `trunk_hidden_dim` proportionally.