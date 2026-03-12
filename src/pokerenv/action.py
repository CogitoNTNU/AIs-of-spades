# pokerenv/action.py
import torch
from dataclasses import dataclass, field
from pokerenv.common import PlayerAction
from pokerenv.observation import Observation


@dataclass
class Action:
    action_type: PlayerAction
    action_tensor: torch.Tensor
    observation: Observation
    bet_amount: float
    bet_tensor: torch.Tensor

    def is_bet(self) -> bool:
        return self.action_type == PlayerAction.BET
