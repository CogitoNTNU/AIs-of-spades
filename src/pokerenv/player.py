from abc import ABC, abstractmethod

from pokerenv.common import PlayerState, PlayerAction
from pokerenv.observation import Observation
from pokerenv.action import Action


class Player(ABC):
    def __init__(self, identifier, name, penalty):
        self.state = PlayerState.ACTIVE
        self.has_acted = False
        self.acted_this_street = False
        self.identifier = identifier
        self.name = name
        self.stack = 0
        self.cards = []
        self.position = 0
        self.all_in = False
        self.bet_this_street = 0
        self.money_in_pot = 0
        self.history = []
        self.hand_rank = 0
        self.pending_penalty = 0
        self.winnings = 0
        self.winnings_for_hh = 0
        self.penalty = penalty

    def __lt__(self, other):
        return self.identifier < other.identifier

    def __gt__(self, other):
        return self.identifier > other.identifier

    def get_reward(self):
        if self.has_acted:
            tmp = self.pending_penalty
            self.pending_penalty = 0
            return tmp + self.winnings
        else:
            return None

    def fold(self):
        self.has_acted = True
        self.acted_this_street = True
        self.state = PlayerState.FOLDED
        self.history.append({"action": PlayerAction.FOLD, "value": 0})

    def _check(self):
        self.has_acted = True
        self.acted_this_street = True
        self.history.append({"action": PlayerAction.CALL, "value": 0})

    def _call(self, amount):
        self.has_acted = True
        self.acted_this_street = True
        amount = amount - self.bet_this_street
        if amount >= self.stack:
            call_size = self.stack
            self.stack = 0
            self.all_in = True
            self.bet_this_street += call_size
            self.money_in_pot += call_size
            self.history.append({"action": PlayerAction.CALL, "value": call_size})
            return call_size
        else:
            self.stack -= amount
            self.bet_this_street += amount
            self.money_in_pot += amount
            self.history.append({"action": PlayerAction.CALL, "value": amount})
            return amount

    def check_or_call(self, amount):
        if amount > self.bet_this_street:
            return self._call(amount)
        else:
            self._check()
            return 0

    def bet(self, amount):
        self.has_acted = True
        self.acted_this_street = True
        if amount == self.stack:
            self.all_in = True
        amount = amount - self.bet_this_street
        self.stack -= amount
        self.bet_this_street += amount
        self.money_in_pot += amount
        self.history.append({"action": PlayerAction.BET, "value": amount})
        return amount

    def punish_invalid_action(self):
        self.pending_penalty += self.penalty

    def finish_street(self):
        self.acted_this_street = False
        self.bet_this_street = 0

    def calculate_hand_rank(self, evaluator, community_cards):
        # treys supports only 5, 6, or 7 cards total.
        # If the board wasn't fully dealt (all-in before river), pad by dealing
        # the remaining cards from a temporary deck — but the real fix is to run
        # out the board in table.py before calling this.
        # This guard prevents a hard crash in edge cases.
        all_cards = self.cards + community_cards
        if len(all_cards) < 5:
            raise ValueError(
                f"calculate_hand_rank called with only {len(all_cards)} cards "
                f"(hole={len(self.cards)}, community={len(community_cards)}). "
                "The board must be fully run out before showdown evaluation."
            )
        self.hand_rank = evaluator.evaluate(self.cards, community_cards)

    def reset(self):
        self.state = PlayerState.ACTIVE
        self.has_acted = False
        self.acted_this_street = False
        self.all_in = False
        self.bet_this_street = 0
        self.money_in_pot = 0
        self.cards = []
        self.history = []
        self.hand_rank = 0
        self.pending_penalty = 0
        self.winnings = 0
        self.winnings_for_hh = 0

    ########################################
    #           Abstract Methods           #
    ########################################
    @abstractmethod
    def get_action(self, observation: Observation) -> Action:
        pass

    @abstractmethod
    def new_hand(self, observation: Observation) -> Action:
        pass
