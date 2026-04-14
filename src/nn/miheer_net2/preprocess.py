from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from pokerenv.observation import Observation

MAX_OPPONENTS = 5
OPP_FEAT_DIM = 8
OBS_SCALAR_DIM = 10


@dataclass
class PreprocessedObservation2:
    cards: torch.Tensor            # [B, 4, 4, 13]
    bets_seq: torch.Tensor         # [B, T, F]
    padding_mask: torch.Tensor     # [B, T] True = padding row
    obs_scalars: torch.Tensor      # [B, OBS_SCALAR_DIM]
    opp_vecs: torch.Tensor         # [B, MAX_OPPONENTS, OPP_FEAT_DIM]
    opp_mask: torch.Tensor         # [B, MAX_OPPONENTS]
    hand_state: Optional[torch.Tensor]
    game_state: Optional[torch.Tensor]
    opp_memory: Optional[torch.Tensor]


def _encode_cards(observation: Observation, device: torch.device) -> torch.Tensor:
    cards = torch.zeros((4, 4, 13), dtype=torch.float32, device=device)

    for card_obs in observation.hand_cards.cards:
        cards[0, int(card_obs.suit), int(card_obs.rank)] = 1.0

    table_mapping = [1, 1, 1, 2, 3]
    for j, card_obs in enumerate(observation.table_cards.cards[:5]):
        cards[table_mapping[j], int(card_obs.suit), int(card_obs.rank)] = 1.0

    return cards


def _encode_bets(observation: Observation, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bets_seq = torch.from_numpy(observation.hand_log).to(device=device, dtype=torch.float32)
    # Rows fully equal to -1 are padding (consistent with env default hand_log init).
    padding_mask = (bets_seq == -1.0).all(dim=-1)
    return bets_seq, padding_mask


def _encode_obs_scalars(observation: Observation, device: torch.device) -> torch.Tensor:
    stack_norm = 200.0
    pot_norm = 400.0
    vec = torch.tensor(
        [
            float(observation.street) / 3.0,
            float(observation.pot) / pot_norm,
            float(observation.bet_to_match) / stack_norm,
            float(observation.minimum_raise) / stack_norm,
            float(observation.player_position) / 5.0,
            float(observation.player_stack) / stack_norm,
            float(observation.player_money_in_pot) / pot_norm,
            float(observation.bet_this_street) / stack_norm,
            float(observation.bet_range.lower_bound) / stack_norm,
            float(observation.bet_range.upper_bound) / stack_norm,
        ],
        dtype=torch.float32,
        device=device,
    )
    return vec


def _encode_opponents(observation: Observation, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    opp_vecs = torch.zeros((MAX_OPPONENTS, OPP_FEAT_DIM), dtype=torch.float32, device=device)
    opp_mask = torch.zeros((MAX_OPPONENTS,), dtype=torch.bool, device=device)

    stack_norm = 200.0
    pot_norm = 400.0
    for i, o in enumerate(observation.others[:MAX_OPPONENTS]):
        opp_vecs[i] = torch.tensor(
            [
                float(o.position) / 5.0,
                float(o.state),
                float(o.stack) / stack_norm,
                float(o.money_in_pot) / pot_norm,
                float(o.bet_this_street) / stack_norm,
                float(o.is_all_in),
                1.0,
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        opp_mask[i] = True

    return opp_vecs, opp_mask


def preprocess_observation(
    observation: Observation,
    *,
    device: Optional[torch.device] = None,
) -> PreprocessedObservation2:
    if device is None:
        device = torch.device("cpu")

    cards = _encode_cards(observation, device)
    bets_seq, padding_mask = _encode_bets(observation, device)
    obs_scalars = _encode_obs_scalars(observation, device)
    opp_vecs, opp_mask = _encode_opponents(observation, device)

    hand_state = None
    game_state = None
    opp_memory = None
    if getattr(observation, "is_replay", False):
        ns = observation.network_internal_state
        hand_state = ns.get("hand", None)
        game_state = ns.get("game", None)
        opp_memory = ns.get("opp_memory", None)
        if hand_state is not None:
            hand_state = torch.as_tensor(hand_state, dtype=torch.float32, device=device)
        if game_state is not None:
            game_state = torch.as_tensor(game_state, dtype=torch.float32, device=device)
        if opp_memory is not None:
            opp_memory = torch.as_tensor(opp_memory, dtype=torch.float32, device=device)

    cards = cards.unsqueeze(0)
    bets_seq = bets_seq.unsqueeze(0)
    padding_mask = padding_mask.unsqueeze(0)
    obs_scalars = obs_scalars.unsqueeze(0)
    opp_vecs = opp_vecs.unsqueeze(0)
    opp_mask = opp_mask.unsqueeze(0)

    if hand_state is not None and hand_state.dim() == 1:
        hand_state = hand_state.unsqueeze(0)
    if game_state is not None and game_state.dim() == 1:
        game_state = game_state.unsqueeze(0)
    if opp_memory is not None and opp_memory.dim() == 1:
        opp_memory = opp_memory.unsqueeze(0)

    return PreprocessedObservation2(
        cards=cards,
        bets_seq=bets_seq,
        padding_mask=padding_mask,
        obs_scalars=obs_scalars,
        opp_vecs=opp_vecs,
        opp_mask=opp_mask,
        hand_state=hand_state,
        game_state=game_state,
        opp_memory=opp_memory,
    )


__all__ = [
    "MAX_OPPONENTS",
    "OPP_FEAT_DIM",
    "OBS_SCALAR_DIM",
    "PreprocessedObservation2",
    "preprocess_observation",
]
