"""
_utils.py

preprocess_observation converts a single Observation into numpy arrays
consumed by SimoNet.  All arrays are on CPU; SimoNet.forward() converts
them to tensors on the model device as needed.

Returning numpy arrays (not torch tensors) avoids torch.multiprocessing's
shared-memory file-handle mechanism (rebuild_storage_filename) when
trajectories are pickled back from worker processes through the pool pipe,
replacing it with standard buffer-protocol pickle which is significantly
cheaper on Windows.

Returned namedtuple PreprocessedObs fields
------------------------------------------
hole_cards   : [2, 2]         int64   (suit_idx, rank_idx) per card
board_cards  : [5, 2]         int64   zero-padded for unseen cards
board_mask   : [5]            bool    True = card is present
bets         : [64, 8]        float32 hand_log (64 actions × 8 features)
obs_scalars  : [10]           float32 see breakdown below
opp_vecs     : [5, 8]         float32 opponent tokens
opp_mask     : [5]            bool    True = opponent present
hand_state   : [1, hand_dim]  float32 | None   from observation.network_internal_state
game_state   : [1, game_dim]  float32 | None

obs_scalars breakdown (10 floats)
----------------------------------
  [0]  street               (0-3, normalised /3)
  [1]  pot                  (normalised /400)
  [2]  bet_to_match         (normalised /200)
  [3]  minimum_raise        (normalised /200)
  [4]  player_position      (0-5, normalised /5)
  [5]  player_stack         (normalised /200)
  [6]  player_money_in_pot  (normalised /400)
  [7]  player_bet_this_street (normalised /200)
  [8]  bet_range_lower      (normalised /200)
  [9]  bet_range_upper      (normalised /200)
"""

from typing import NamedTuple, Optional
import numpy as np
from pokerenv.observation import Observation

MAX_OPPONENTS = 5
OPP_FEAT_DIM = 8  # features per opponent slot


class PreprocessedObs(NamedTuple):
    hole_cards: np.ndarray   # [2, 2]       int64
    board_cards: np.ndarray  # [5, 2]       int64
    board_mask: np.ndarray   # [5]          bool
    bets: np.ndarray         # [64, 8]      float32
    obs_scalars: np.ndarray  # [10]         float32
    opp_vecs: np.ndarray     # [5, 8]       float32
    opp_mask: np.ndarray     # [5]          bool
    hand_state: Optional[np.ndarray]        # [1, hand_dim] float32 | None
    game_state: Optional[np.ndarray]        # [1, game_dim] float32 | None


def preprocess_observation(observation: Observation) -> PreprocessedObs:

    # ── HOLE CARDS ────────────────────────────────────────────────────
    hole = np.zeros((2, 2), dtype=np.int64)
    for i, c in enumerate(observation.hand_cards.cards[:2]):
        hole[i, 0] = int(c.suit)
        hole[i, 1] = int(c.rank)

    # ── BOARD CARDS ───────────────────────────────────────────────────
    board = np.zeros((5, 2), dtype=np.int64)
    board_mask = np.zeros(5, dtype=bool)

    street = int(observation.street)
    n_board = {0: 0, 1: 3, 2: 4, 3: 5}.get(street, 0)

    for i, c in enumerate(observation.table_cards.cards[:n_board]):
        board[i, 0] = int(c.suit)
        board[i, 1] = int(c.rank)
        board_mask[i] = True

    # ── HAND LOG (bet history) ────────────────────────────────────────
    # hand_log shape: [64, 8] — kept as-is for BetsNN (no flatten)
    bets = observation.hand_log.astype(np.float32)

    # ── OBS SCALARS (self features, 10 values) ────────────────────────
    STACK_NORM = 200.0
    POT_NORM = 400.0

    self_feats = np.array(
        [
            float(observation.street) / 3.0,
            float(observation.pot) / POT_NORM,
            float(observation.bet_to_match) / STACK_NORM,
            float(observation.minimum_raise) / STACK_NORM,
            float(observation.player_position) / 5.0,
            float(observation.player_stack) / STACK_NORM,
            float(observation.player_money_in_pot) / POT_NORM,
            float(observation.bet_this_street) / STACK_NORM,
            float(observation.bet_range.lower_bound) / STACK_NORM,
            float(observation.bet_range.upper_bound) / STACK_NORM,
        ],
        dtype=np.float32,
    )

    # ── OPPONENT TOKENS ───────────────────────────────────────────────
    opp_vecs = np.zeros((MAX_OPPONENTS, OPP_FEAT_DIM), dtype=np.float32)
    opp_mask = np.zeros(MAX_OPPONENTS, dtype=bool)

    for i, o in enumerate(observation.others[:MAX_OPPONENTS]):
        opp_vecs[i] = [
            float(o.position),
            float(o.state),
            float(o.stack),
            float(o.money_in_pot),
            float(o.bet_this_street),
            float(o.is_all_in),
            1.0,  # is_present flag
            0.0,  # reserved / padding
        ]
        opp_mask[i] = True

    # ── INTERNAL STATE (replay) ───────────────────────────────────────
    hand_state = None
    game_state = None
    if observation.is_replay:
        hs = observation.network_internal_state["hand"]
        gs = observation.network_internal_state["game"]
        hand_state = hs.detach().cpu().numpy() if hasattr(hs, "detach") else np.asarray(hs, dtype=np.float32)
        game_state = gs.detach().cpu().numpy() if hasattr(gs, "detach") else np.asarray(gs, dtype=np.float32)

    return PreprocessedObs(
        hole_cards=hole,
        board_cards=board,
        board_mask=board_mask,
        bets=bets,
        obs_scalars=self_feats,
        opp_vecs=opp_vecs,
        opp_mask=opp_mask,
        hand_state=hand_state,
        game_state=game_state,
    )
