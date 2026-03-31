# pokerenv/action.py
from dataclasses import dataclass
from pokerenv.common import PlayerAction
from pokerenv.observation import Observation


@dataclass
class Action:
    action_type: PlayerAction
    observation: Observation
    bet_amount: float       # actual chip value sent to the table
    bet_normalized: float   # sampled value in [0, 1] used for the continuous loss

    def is_bet(self) -> bool:
        return self.action_type == PlayerAction.BET
