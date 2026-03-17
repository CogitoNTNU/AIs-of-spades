# ui/ai_player.py

import math
import torch
import torch.distributions as D

from pokerenv.observation import Observation
from pokerenv.common import PlayerAction
from pokerenv.action import Action
from .ui_player import UIPlayer


class AIPlayer(UIPlayer):

    def __init__(self, seat: int, name: str, model_class, weights_path: str):
        super().__init__(seat, name)
        self._model = model_class()
        self._model.initialize_internal_state()
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def get_action(self, obs: Observation) -> Action:
        with torch.no_grad():
            action_logits, bet_mean, bet_std = self._model.forward(obs)

        discrete_dist = D.Categorical(logits=action_logits)
        d = discrete_dist.sample()

        continuous_dist = D.Normal(bet_mean, bet_std)
        bet_sample = continuous_dist.sample().clamp(0.0, 1.0)

        bet_value = (
            bet_sample.item() * (obs.bet_range.upper_bound - obs.bet_range.lower_bound)
            + obs.bet_range.lower_bound
        )

        return Action(
            action_type=PlayerAction(d.item()),
            action_tensor=d,
            observation=obs,
            bet_amount=bet_value,
            bet_tensor=bet_sample,
        )

    def new_hand(self):
        self._model.new_hand()

    def reset(self):
        super().reset()
        self._model.initialize_internal_state()

    @property
    def is_connected(self) -> bool:
        return True

    def attach_websocket(self, websocket, loop):
        pass

    def detach_websocket(self):
        pass

    def receive_action_from_client(self, msg: dict):
        pass
