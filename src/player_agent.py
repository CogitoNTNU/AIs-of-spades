from pokerenv.observation import Observation
from pokerenv.common import Action
from pokerenv.player import Player
from nn.nn import PokerNet
import numpy as np
from scipy.stats import norm


class PlayerAgent(Player):
    def __init__(self, identifier, name, penalty, nn: PokerNet):
        super().__init__(identifier, name, penalty)
        self.nn = nn

    def get_action(self, observation) -> Action:
        """
        Input: an object of Observation class
        Output: an Action
        """
        # Get observation and feed it to the network
        action_logits, bet_mean, bet_variance = self.nn.forward(observation)

        # Sample action according to the distribution given by the network logits
        d = np.random.choice(len(action_logits), p=action_logits)

        # Sample bet amount from the predicted distribution
        bet_value = np.random.normal(bet_mean.item(), np.sqrt(bet_variance.item()))
        bet_prob = norm.pdf(bet_value, bet_mean.item(), np.sqrt(bet_variance.item()))

        # scale bet_value according to the bet range in the observation
        bet_value = (
            bet_value
            * (observation.bet_range.upper_bound - observation.bet_range.lower_bound)
            + observation.bet_range.lower_bound
        )

        return Action(d, action_logits[d], bet_value, bet_prob)

    def reset(self):
        pass
