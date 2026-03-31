# player_agent.py
import torch.distributions as D

from pokerenv.observation import Observation
from pokerenv.common import PlayerAction
from pokerenv.action import Action
from pokerenv.player import Player
from nn.poker_net import PokerNet


class PlayerAgent(Player):
    def __init__(self, identifier, name, penalty, nn: PokerNet):
        super().__init__(identifier, name, penalty)
        self.nn = nn

    def get_action(self, observation: Observation) -> Action:
        action_logits, bet_mean, bet_std = self.nn.forward(observation)

        discrete_dist = D.Categorical(logits=action_logits)
        d = discrete_dist.sample()

        continuous_dist = D.Normal(bet_mean, bet_std)
        bet_sample = continuous_dist.sample().clamp(0.0, 1.0)

        bet_value = (
            bet_sample.item()
            * (observation.bet_range.upper_bound - observation.bet_range.lower_bound)
            + observation.bet_range.lower_bound
        )

        return Action(
            action_type=PlayerAction(int(d.item())),
            observation=observation,
            bet_amount=bet_value,
            bet_normalized=float(bet_sample.item()),
        )

    def new_hand(self):
        self.nn.new_hand()

    def reset(self):
        super().reset()
