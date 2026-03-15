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
        valid_actions = [PlayerAction.FOLD, PlayerAction.BET, PlayerAction.CALL]
        valid_bet_range = [
            max(self.bet_to_match + self.minimum_raise, 1.0),
            player.stack,
        ]

        others_active = [
            p
            for p in players
            if p.state is PlayerState.ACTIVE and not p.all_in and p is not player
        ]

        if self.bet_to_match == 0:
            valid_actions.remove(PlayerAction.FOLD)

        if (
            player.stack < max(self.bet_to_match + self.minimum_raise, 1.0)
            or len(others_active) == 0
        ):
            valid_bet_range = [0.0, 0.0]
            if PlayerAction.BET in valid_actions:
                valid_actions.remove(PlayerAction.BET)

        # print(
        #     f"DEBUG betting: stack={player.stack}, bet_to_match={self.bet_to_match}, "
        #     f"min_raise={self.minimum_raise}, min_bet={max(self.bet_to_match + self.minimum_raise, 1.0)}, "
        #     f"others_active={len(others_active)}"
        # )

        return {"actions_list": valid_actions, "bet_range": valid_bet_range}

    def is_action_valid(self, player, action, valid_actions: dict) -> tuple:
        """
        Returns (is_valid: bool, fallback: PlayerAction | None).
        Table is responsible for applying the fallback — BettingManager only decides.
        """
        action_list = valid_actions["actions_list"]
        bet_range = valid_actions["bet_range"]

        if action.action_type not in action_list:
            if PlayerAction.FOLD in action_list:
                return False, PlayerAction.FOLD
            if PlayerAction.CALL in action_list:
                return False, PlayerAction.CALL
            raise Exception("No valid fallback action found in valid_actions")

        if action.action_type is PlayerAction.BET:
            out_of_range = not (
                approx_lte(bet_range[0], action.bet_amount)
                and approx_lte(action.bet_amount, bet_range[1])
            )
            if out_of_range or approx_gt(action.bet_amount, player.stack):
                return False, (
                    PlayerAction.FOLD
                    if PlayerAction.FOLD in action_list
                    else PlayerAction.CALL
                )

        return True, None
