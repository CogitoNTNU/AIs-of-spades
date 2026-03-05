from enum import IntEnum, Enum


class GameState(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class PlayerState(Enum):
    FOLDED = 0
    ACTIVE = 1


class PlayerAction(IntEnum):
    FOLD = 0
    BET = 1
    CALL = 2


class TablePosition(IntEnum):
    SB = 0
    BB = 1


class Action:
    def __init__(self, action_type, action_prob, bet_amount, bet_prob):
        self.action_type = action_type
        self.action_prob = action_prob
        self.bet_amount = bet_amount
        self.bet_prob = bet_prob


action_list = [PlayerAction.FOLD, PlayerAction.BET, PlayerAction.CALL]
