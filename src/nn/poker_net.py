from abc import ABC, abstractmethod
from typing import Tuple
from pokerenv.observation import Observation
import torch
import torch.nn as nn


class PokerNet(nn.Module, ABC):
    """
    Interface for all Poker neural networks.
    """

    @abstractmethod
    def forward(
        self,
        observation: Observation,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Must return:
            - action_logits
            - value
        """
        pass

    @abstractmethod
    def initialize_internal_state(self) -> None:
        """
        Must return:
            - action_logits
            - value
        """
        pass

    # @abstractmethod
    # def signal_hand_end(self):
    #     """
    #     Signal the end of a hand and provide the final reward.
    #     This can be used to update internal state or perform learning updates.
    #     """
    #     pass

    # @abstractmethod
    # def signal_game_end(self, reward: float):
    #     """
    #     Signal the end of a game and provide the final reward.
    #     This can be used to update internal state or perform learning updates.
    #     """
    #     pass

    # more functions can be added

    # next_hand, clearing part of the internal state, etc.
    # next_game, clearing the entire internal state, etc.
    # reset function to clear the entire internal state, etc.
