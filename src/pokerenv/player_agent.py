# player_agent.py
import numpy as np
import torch
import torch.distributions as D

from pokerenv.observation import Observation
from pokerenv.common import PlayerAction
from pokerenv.action import Action
from pokerenv.player import Player
from nn.nn import PokerNet


class PlayerAgent(Player):
    def __init__(self, identifier, name, penalty, nn: PokerNet):
        super().__init__(identifier, name, penalty)
        self.nn = nn

    def get_action(self, observation: Observation) -> Action:
        """
        Runs a forward pass and samples an action.

        Returns an Action whose log_prob fields are PyTorch tensors
        still attached to the computation graph, so that
        loss.backward() can propagate gradients to the network.
        """
        action_logits, bet_mean, bet_std = self.nn.forward(observation)

        # --- Discrete action ---
        # action_logits: torch.Tensor of shape (n_actions,), after softmax
        discrete_dist = D.Categorical(logits=action_logits)
        d = discrete_dist.sample()  # scalar tensor
        log_p_discrete = discrete_dist.log_prob(d)  # scalar tensor, attached to graph

        # --- Continuous bet amount ---
        # bet_mean, bet_std: scalar tensors from the network
        continuous_dist = D.Normal(bet_mean, bet_std)
        bet_sample = continuous_dist.sample()  # scalar tensor
        log_p_continuous = continuous_dist.log_prob(
            bet_sample
        )  # scalar tensor, attached to graph

        # Scale bet into the valid range from the observation
        bet_value = (
            bet_sample.item()
            * (observation.bet_range.upper_bound - observation.bet_range.lower_bound)
            + observation.bet_range.lower_bound
        )

        # Detach the int/float values used by the environment,
        # but keep the tensor log_probs alive for the learning loop
        action_type = d.item()

        return Action(
            action_type=PlayerAction(action_type),
            log_p_discrete=log_p_discrete,  # tensor — kept for backward()
            bet_amount=bet_value,  # float — used by Table
            log_p_continuous=log_p_continuous,  # tensor — kept for backward()
        )

    def reset(self):
        super().reset()
        self.nn.initialize_internal_state()
