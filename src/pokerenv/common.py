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
    UTG = 2
    UTG_1 = 3
    UTG_2 = 4
    BTN = 5

    @classmethod
    def label(cls, pos: int, n_players: int) -> str:
        if pos == 0:
            return "SB"
        if pos == 1:
            return "BB"
        if pos == n_players - 1:
            return "BTN"
        if pos == 2:
            return "UTG"
        return "UTG+%d" % (pos - 2)

    @classmethod
    def hh_label(cls, pos: int, n_players: int) -> str:
        """Long-form label for hand history files."""
        if pos == 0:
            return "small blind"
        if pos == 1:
            return "big blind"
        if pos == n_players - 1:
            return "button"
        if pos == 2:
            return "UTG"
        return "UTG+%d" % (pos - 2)
