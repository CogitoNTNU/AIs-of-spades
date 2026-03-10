import numpy as np


class CardObservation:
    def __init__(self, card):
        self.suit = card[0]
        self.rank = card[1]


class HandObservation:
    def __init__(self, cards):
        self.cards = [
            CardObservation(cards[i : i + 2]) for i in range(0, len(cards) - 1, 2)
        ]


class TableObservation:
    def __init__(self, cards):
        self.cards = [
            CardObservation(cards[i : i + 2]) for i in range(0, len(cards) - 1, 2)
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

    def can_check(self):
        return self.actions[0] == 1

    def can_fold(self):
        return self.actions[1] == 1

    def can_bet(self):
        return self.actions[2] == 1

    def can_call(self):
        return self.actions[3] == 1


class BetRangeObservation:
    def __init__(self, bounds):
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]


class Observation:
    def __init__(self, obs_matrix, hand_log):
        self.player_identifier = obs_matrix[0]
        self.actions = ActionsObservation(obs_matrix[1:5])
        self.bet_range = BetRangeObservation(obs_matrix[5:7])
        self.player_position = obs_matrix[8]
        self.hand_cards = HandObservation(obs_matrix[9:12])
        self.player_stack = obs_matrix[12]
        self.player_money_in_pot = obs_matrix[13]
        self.bet_this_street = obs_matrix[14]
        self.street = obs_matrix[15]
        self.table_cards = TableObservation(obs_matrix[16:26])
        self.pot = obs_matrix[26]
        self.bet_to_match = obs_matrix[27]
        self.minimum_raise = obs_matrix[28]

        self.others = [
            OtherPlayerObservation(obs_matrix[29 + i : 29 + i + 6])
            for i in range(0, (len(obs_matrix) - 29) // 6, 6)
        ]
        self.hand_log = hand_log

    @staticmethod
    def empty():
        return Observation(np.zeros(59, dtype=np.float32), np.full((32, 4), -1), dtype=np.float32)
