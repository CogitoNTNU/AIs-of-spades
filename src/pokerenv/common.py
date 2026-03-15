from enum import IntEnum, Enum


class GameState(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class PlayerState(Enum):
    FOLDED = 0
    ACTIVE = 1
    OUT = 2


class PlayerAction(IntEnum):
    FOLD = 0
    BET = 1
    CALL = 2


class TablePosition(IntEnum):
    SB = 0
    BB = 1
