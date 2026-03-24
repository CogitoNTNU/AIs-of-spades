"""
_utils.py

preprocess_observation converts a single Observation into the tensors
consumed by SimoNet.  All tensors are on CPU; SimoNet.forward() moves
them to the model device as needed.

Returned namedtuple PreprocessedObs fields
------------------------------------------
hole_cards   : [2, 2]         int   (suit_idx, rank_idx) per card
board_cards  : [5, 2]         int   zero-padded for unseen cards
board_mask   : [5]            bool  True = card is present
bets         : [512]          float flattened hand_log (64×8)
obs_scalars  : [48]           float see breakdown below
hand_state   : [hand_dim] | None   from observation.network_internal_state
game_state   : [game_dim] | None   from observation.network_internal_state

obs_scalars breakdown (48 floats)
----------------------------------
  [0]  street               (0-3)
  [1]  pot
  [2]  bet_to_match
  [3]  minimum_raise
  [4]  player_position      (0-5)
  [5]  player_stack
  [6]  player_money_in_pot
  [7]  player_bet_this_street
  [8]  bet_range_lower
  [9]  bet_range_upper
  [10..49]  5 opponents × 8 features each:
            position, state, stack, money_in_pot,
            bet_this_street, is_all_in, is_present(1.0), padding(0.0)

opponent tokens breakdown (for TransformerTrunk)
-------------------------------------------------
  Built separately as opp_vecs [5, 8] and opp_mask [5] bool.
  Absent seats → zero vector + mask=False.
"""

from typing import NamedTuple, Optional
import torch
from pokerenv.observation import Observation

MAX_OPPONENTS = 5
OPP_FEAT_DIM = 8  # features per opponent slot


class PreprocessedObs(NamedTuple):
    hole_cards: torch.Tensor  # [2, 2]   int
    board_cards: torch.Tensor  # [5, 2]   int
    board_mask: torch.Tensor  # [5]      bool
    bets: torch.Tensor  # [512]    float
    obs_scalars: torch.Tensor  # [48]     float  (self only, no opp)
    opp_vecs: torch.Tensor  # [5, 8]   float  opponent tokens
    opp_mask: torch.Tensor  # [5]      bool   True = present
    hand_state: Optional[torch.Tensor]
    game_state: Optional[torch.Tensor]


def preprocess_observation(observation: Observation) -> PreprocessedObs:

    # ── HOLE CARDS ────────────────────────────────────────────────────
    hole = torch.zeros(2, 2, dtype=torch.long)
    for i, c in enumerate(observation.hand_cards.cards[:2]):
        hole[i, 0] = int(c.suit)
        hole[i, 1] = int(c.rank)

    # ── BOARD CARDS ───────────────────────────────────────────────────
    board = torch.zeros(5, 2, dtype=torch.long)
    board_mask = torch.zeros(5, dtype=torch.bool)

    street = int(observation.street)
    n_board = {0: 0, 1: 3, 2: 4, 3: 5}.get(street, 0)

    for i, c in enumerate(observation.table_cards.cards[:n_board]):
        board[i, 0] = int(c.suit)
        board[i, 1] = int(c.rank)
        board_mask[i] = True

    # ── HAND LOG (bet history) ────────────────────────────────────────
    # hand_log shape: [64, 8]  → flatten → [512]
    bets = torch.from_numpy(observation.hand_log).float().flatten()

    # ── OBS SCALARS (self features, 10 values) ────────────────────────
    self_feats = torch.tensor(
        [
            float(observation.street),
            float(observation.pot),
            float(observation.bet_to_match),
            float(observation.minimum_raise),
            float(observation.player_position),
            float(observation.player_stack),
            float(observation.player_money_in_pot),
            float(observation.bet_this_street),
            float(observation.bet_range.lower_bound),
            float(observation.bet_range.upper_bound),
        ],
        dtype=torch.float32,
    )

    # ── OPPONENT TOKENS ───────────────────────────────────────────────
    opp_vecs = torch.zeros(MAX_OPPONENTS, OPP_FEAT_DIM, dtype=torch.float32)
    opp_mask = torch.zeros(MAX_OPPONENTS, dtype=torch.bool)

    actual_opps = observation.others[:MAX_OPPONENTS]

    for i, o in enumerate(actual_opps):
        opp_vecs[i] = torch.tensor(
            [
                float(o.position),
                float(o.state),
                float(o.stack),
                float(o.money_in_pot),
                float(o.bet_this_street),
                float(o.is_all_in),
                1.0,  # is_present flag
                0.0,  # reserved / padding
            ],
            dtype=torch.float32,
        )
        opp_mask[i] = True

    # ── INTERNAL STATE (replay) ───────────────────────────────────────
    hand_state = None
    game_state = None
    if observation.is_replay:
        hand_state = observation.network_internal_state["hand"]
        game_state = observation.network_internal_state["game"]

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
