"""
Preprocessing utilities for MiheerNet.

This module prepares observation fields for the modular MiheerNet components:
- CardsEncoder (CNN): expects a float tensor shaped [B, 4, 4, 13].
- BetsTransformer (Transformer): expects a float tensor shaped [B, T, F] (e.g., [B,64,8]) or [B, F_flat] (e.g., [B,512]).
- StateMLP (MLP): expects hand/game internal state tensors shaped [B, hand_dim] / [B, game_dim].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from pokerenv.observation import Observation


@dataclass
class PreprocessedObservation:
    """
    Container for preprocessed tensors.

    Attributes:
        cards: Tensor [B, 4, 4, 13]
        bets_seq: Tensor [B, T, F] (unflattened hand log)
        bets_flat: Tensor [B, T*F] (flattened hand log)
        hand_state: Optional tensor [B, H]
        game_state: Optional tensor [B, G]
        padding_mask: Optional bool tensor [B, T] (True = padding)
    """
    cards: torch.Tensor
    bets_seq: torch.Tensor
    bets_flat: torch.Tensor
    hand_state: Optional[torch.Tensor]
    game_state: Optional[torch.Tensor]
    padding_mask: Optional[torch.Tensor]


def _encode_cards(observation: Observation, device: torch.device) -> torch.Tensor:
    """
    Encode hand and table cards into a dense one-hot grid [4, 4, 13].
    Slots: 0=hand, 1-3=community (flop/turn/river mapping).
    """
    card_tensor = torch.zeros((4, 4, 13), dtype=torch.float32, device=device)

    for card_obs in observation.hand_cards.cards:
        suit_idx = int(card_obs.suit)
        rank_idx = int(card_obs.rank)
        card_tensor[0, suit_idx, rank_idx] = 1.0

    table_mapping = [1, 1, 1, 2, 3]  # first 3 flop slots share index 1
    for j, card_obs in enumerate(observation.table_cards.cards):
        suit_idx = int(card_obs.suit)
        rank_idx = int(card_obs.rank)
        slot = table_mapping[j]
        card_tensor[slot, suit_idx, rank_idx] = 1.0

    return card_tensor


def _encode_bets(observation: Observation, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert hand_log numpy array [T, F] (expected 64 x 8) to:
    - bets_seq: torch.Tensor [T, F]
    - bets_flat: torch.Tensor [T*F]
    """
    bets_seq = torch.from_numpy(observation.hand_log).float().to(device)
    bets_flat = bets_seq.flatten()
    return bets_seq, bets_flat


def preprocess_observation(
    observation: Observation,
    *,
    device: Optional[torch.device] = None,
    return_padding_mask: bool = False,
) -> PreprocessedObservation:
    """
    Convert Observation into tensors compatible with MiheerNet branches.

    Args:
        observation: pokerenv Observation.
        device: torch device; if None, inferred from available tensors.
        return_padding_mask: If True, creates a padding mask of zeros
                             (useful for variable-length extensions).

    Returns:
        PreprocessedObservation with card grid, betting sequence/flat,
        optional internal states when replaying, and optional padding mask.
    """
    if device is None:
        device = torch.device("cpu")

    cards = _encode_cards(observation, device)
    bets_seq, bets_flat = _encode_bets(observation, device)

    padding_mask = None
    if return_padding_mask:
        # Current logs are dense; mask is all False (no padding).
        padding_mask = torch.zeros((bets_seq.shape[0],), dtype=torch.bool, device=device)

    hand_state = None
    game_state = None
    if getattr(observation, "is_replay", False):
        hand_state = observation.network_internal_state.get("hand", None)
        game_state = observation.network_internal_state.get("game", None)
        if hand_state is not None:
            hand_state = torch.as_tensor(hand_state, device=device, dtype=torch.float32)
        if game_state is not None:
            game_state = torch.as_tensor(game_state, device=device, dtype=torch.float32)

    # Add batch dimension where appropriate
    cards = cards.unsqueeze(0)
    bets_seq = bets_seq.unsqueeze(0)
    bets_flat = bets_flat.unsqueeze(0)
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(0)
    if hand_state is not None and hand_state.dim() == 1:
        hand_state = hand_state.unsqueeze(0)
    if game_state is not None and game_state.dim() == 1:
        game_state = game_state.unsqueeze(0)

    return PreprocessedObservation(
        cards=cards,
        bets_seq=bets_seq,
        bets_flat=bets_flat,
        hand_state=hand_state,
        game_state=game_state,
        padding_mask=padding_mask,
    )


__all__ = ["PreprocessedObservation", "preprocess_observation"]
