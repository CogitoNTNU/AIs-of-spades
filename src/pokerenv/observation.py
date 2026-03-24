import numpy as np
import math
from pokerenv.common import PlayerAction


class CardObservation:
    def __init__(self, card):
        self.suit = math.log2(card[0]) if card[0] > 0 else 0
        self.rank = card[1]


class HandObservation:
    def __init__(self, cards):
        self.cards = [
            CardObservation(cards[i : i + 2]) for i in range(0, len(cards), 2)
        ]


class TableObservation:
    def __init__(self, cards):
        self.cards = [
            CardObservation(cards[i : i + 2]) for i in range(0, len(cards), 2)
        ]


class OtherPlayerObservation:
    def __init__(self, obs_matrix):
        self.position = obs_matrix[0]
        self.state = obs_matrix[1]
        self.stack = obs_matrix[2]
        self.money_in_pot = obs_matrix[3]
        self.bet_this_street = obs_matrix[4]
        self.is_all_in = obs_matrix[5]


class ActionsObservation:
    def __init__(self, actions):
        self.actions = actions

    def can_fold(self):
        return self.actions[PlayerAction.FOLD] == 1

    def can_bet(self):
        return self.actions[PlayerAction.BET] == 1

    def can_call(self):
        return self.actions[PlayerAction.CALL] == 1


class BetRangeObservation:
    def __init__(self, bounds):
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]


class Observation:
    def __init__(self, obs_matrix, hand_log):
        self.player_identifier = obs_matrix[0]
        self.actions = ActionsObservation(obs_matrix[1:4])
        self.bet_range = BetRangeObservation(obs_matrix[4:6])
        self.player_position = obs_matrix[6]
        self.hand_cards = HandObservation(obs_matrix[7:11])
        self.player_stack = obs_matrix[11]
        self.player_money_in_pot = obs_matrix[12]
        self.bet_this_street = obs_matrix[13]
        self.street = obs_matrix[14]
        self.table_cards = TableObservation(obs_matrix[15:25])
        self.pot = obs_matrix[25]
        self.bet_to_match = obs_matrix[26]
        self.minimum_raise = obs_matrix[27]

        self.others = [
            OtherPlayerObservation(obs_matrix[28 + i * 6 : 28 + i * 6 + 6])
            for i in range((len(obs_matrix) - 28) // 6)
        ]
        self.hand_log = hand_log
        self.is_replay = False

    def add_network_internal_state(self, network_internal_state):
        self.is_replay = True
        self.network_internal_state = network_internal_state

    @staticmethod
    def empty():
        obs_array = np.zeros(58, dtype=np.float32)
        hand_log = np.full((64, 8), -1.0, dtype=np.float32)
        return Observation(obs_array, hand_log)

    def __str__(self) -> str:
        hand = [(c.suit, c.rank) for c in self.hand_cards.cards]
        table = [(c.suit, c.rank) for c in self.table_cards.cards]
        actions = {
            "fold": self.actions.can_fold(),
            "bet": self.actions.can_bet(),
            "call": self.actions.can_call(),
        }
        others_info = []
        for i, o in enumerate(self.others):
            others_info.append(
                f"Player {i}: pos={o.position}, state={o.state}, "
                f"stack={o.stack}, money_in_pot={o.money_in_pot}, "
                f"bet_this_street={o.bet_this_street}, all_in={o.is_all_in}"
            )

        return (
            f"--------------------- OBS -----------------------\n"
            f"Observation(player_id={self.player_identifier}, "
            f"position={self.player_position}, stack={self.player_stack}, "
            f"money_in_pot={self.player_money_in_pot}, bet_this_street={self.bet_this_street}, "
            f"street={self.street}, pot={self.pot}, bet_to_match={self.bet_to_match}, "
            f"minimum_raise={self.minimum_raise},\n"
            f"  hand_cards={hand},\n"
            f"  table_cards={table},\n"
            f"  actions={actions},\n"
            f"  bet_range=({self.bet_range.lower_bound}, {self.bet_range.upper_bound}),\n"
            f"  others=[\n    " + "\n    ".join(others_info) + "\n  ],\n"
            f"  hand_log_shape={self.hand_log.shape})"
            f"-------------------------------------------------\n"
        )
