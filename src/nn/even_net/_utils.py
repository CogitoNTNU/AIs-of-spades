import torch
from pokerenv.observation import Observation
from typing import Tuple


def preprocess_observation(
    observation: Observation,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    Converts a single Observation object into card and betting tensors.

    Returns:
        cards: [4, 4, 13] tensor
        bets: [128] tensor (flattened hand log)
    """

    # --------- CARDS ---------
    card_tensor = torch.zeros((4, 4, 13), dtype=torch.float32)

    # Encode hand cards
    for card_obs in observation.hand_cards.cards:
        suit_idx = int(card_obs.suit)
        rank_idx = int(card_obs.rank)
        card_tensor[0, suit_idx, rank_idx] = 1.0

    table_mapping = [1, 1, 1, 2, 3]
    for j, card_obs in enumerate(observation.table_cards.cards):
        slot = table_mapping[j]
        suit_idx = int(card_obs.suit)
        rank_idx = int(card_obs.rank)
        card_tensor[slot, suit_idx, rank_idx] = 1.0

    # --------- BET HISTORY ---------

    # convert numpy log to tensor
    bets_tensor = torch.from_numpy(observation.hand_log).float()

    # flatten [32,4] → [128]
    bets_tensor = bets_tensor.flatten()

    network_internal_state_hand = None
    network_internal_state_game = None

    if observation.is_replay:
        network_internal_state_hand = observation.network_internal_state["hand"]
        network_internal_state_game = observation.network_internal_state["game"]

    return (
        card_tensor,
        bets_tensor,
        network_internal_state_hand,
        network_internal_state_game,
    )
