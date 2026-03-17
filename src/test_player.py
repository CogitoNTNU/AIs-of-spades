# test_player.py
from pokerenv.player import Player
from training.common import PlayerAction
from pokerenv.action import Action
from pokerenv.observation import Observation
import torch


class TestPlayerAgent(Player):
    """
    A human-controlled player for testing Table behavior.
    Actions are injected externally via set_next_action(),
    making it usable from a CLI, a test script, or a UI.
    """

    def __init__(self, identifier, name):
        super().__init__(identifier, name, penalty=0)
        self._next_action = None

    def set_next_action(self, action: Action):
        """Called externally (by TestTable or a UI) before the environment calls get_action."""
        self._next_action = action

    def get_action(self, observation: Observation) -> Action:
        if self._next_action is None:
            raise RuntimeError(
                "No action set for player '%s'. Call set_next_action() first."
                % self.name
            )
        action = self._next_action
        self._next_action = None
        return action

    def make_fold(self) -> Action:
        return Action(action_type=PlayerAction.FOLD, log_p_discrete=torch.tensor(1.0))

    def make_call(self) -> Action:
        return Action(action_type=PlayerAction.CALL, log_p_discrete=torch.tensor(1.0))

    def make_bet(self, amount: float) -> Action:
        return Action(
            action_type=PlayerAction.BET,
            log_p_discrete=torch.tensor(1.0),
            bet_amount=amount,
            log_p_continuous=torch.tensor(1.0),
        )

    def reset(self):
        super().reset()
        self._next_action = None
