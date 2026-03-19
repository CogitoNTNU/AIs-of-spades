# table_engine/betting_manager.py
from pokerenv.common import PlayerState, PlayerAction
from pokerenv.utils import approx_gt, approx_lte

BB = 5


class BettingManager:
    """
    Handles everything related to bets on a single street.
    Does NOT touch the pot or active_players count — those are Table's responsibility.
    """

    def __init__(self):
        self.bet_to_match = 0.0
        self.minimum_raise = 0.0
        self.last_bet_placed_by = None

    def reset(self):
        self.bet_to_match = 0.0
        self.minimum_raise = 0.0
        self.last_bet_placed_by = None

    def change_bet_to_match(self, new_amount: float):
        self.minimum_raise = new_amount - self.bet_to_match
        self.bet_to_match = new_amount

    def get_valid_actions(self, player, players: list) -> dict:
        others_active = [
            p
            for p in players
            if p.state is PlayerState.ACTIVE and not p.all_in and p is not player
        ]

        min_bet = max(self.bet_to_match + self.minimum_raise, 1.0)
        can_bet = player.stack >= min_bet and len(others_active) > 0

        valid_actions = [PlayerAction.FOLD, PlayerAction.CALL]
        if can_bet:
            valid_actions.append(PlayerAction.BET)

        valid_bet_range = [min_bet, player.stack] if can_bet else [0.0, 0.0]

        return {"actions_list": valid_actions, "bet_range": valid_bet_range}
