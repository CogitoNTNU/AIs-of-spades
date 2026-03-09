# pokerenv/action.py
import torch
from dataclasses import dataclass, field
from pokerenv.common import PlayerAction


@dataclass
class Action:
    action_type: PlayerAction
    log_p_discrete: torch.Tensor
    bet_amount: float = 0.0
    log_p_continuous: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def is_bet(self) -> bool:
        return self.action_type == PlayerAction.BET
