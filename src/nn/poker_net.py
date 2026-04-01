from abc import ABC, abstractmethod
from typing import Any, Tuple
from pokerenv.observation import Observation
import torch
import torch.nn as nn


class PokerNet(nn.Module, ABC):
    """
    Abstract interface for all poker policy networks.

    Subclasses must implement:
        - forward          : single-step inference during simulation (live state)
        - forward_batch    : batched inference during training replay (no state update)
        - preprocess       : converts a raw Observation into the network's internal format
        - initialize_internal_state : allocates recurrent state before a new game
        - new_hand         : resets per-hand recurrent state between hands
    """

    @abstractmethod
    def forward(
        self,
        observation: Observation,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step forward pass during simulation.
        Updates internal recurrent state as a side effect.

        Returns
        -------
        action_logits : [1, 3]   fold / call / bet
        bet_mean      : [1, 1]   in (0, 1)
        bet_std       : [1, 1]   > 0
        """
        pass

    @abstractmethod
    def forward_batch(
        self,
        trajectory: list,  # list of (preprocessed_obs, Action)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched forward pass over a full trajectory for training replay.
        Must NOT update internal recurrent state.
        Receives pre-processed observations as returned by preprocess().

        Returns
        -------
        action_logits : [N, 3]
        bet_mean      : [N, 1]
        bet_std       : [N, 1]
        """
        pass

    @abstractmethod
    def preprocess(self, observation: Observation) -> Any:
        """
        Converts a raw Observation into the network's internal input format.
        Called inside the simulation worker (CPU, no grad).
        Must not access or modify internal recurrent state.
        The returned object is stored in the trajectory and later passed to forward_batch().
        """
        pass

    @abstractmethod
    def initialize_internal_state(self) -> None:
        """
        Allocates and zeroes all recurrent state tensors.
        Must be called once before the first forward() of a new game session.
        """
        pass

    @abstractmethod
    def new_hand(self) -> None:
        """
        Resets per-hand recurrent state (hand_state).
        Cross-hand state (game_state) is preserved.
        Must be called between hands.
        """
        pass

    # ---------------------------------------------------------------------------
    # Hooks for future extensions — not yet enforced by the interface.
    #
    # signal_hand_end(reward)  : called at hand end with the hand reward
    # signal_game_end(reward)  : called at game end with the total reward
    # ---------------------------------------------------------------------------
