from nn.poker_net import PokerNet
from pokerenv.observation import Observation
from typing import Tuple
import torch
import torch.nn as nn

class FedeNet(PokerNet):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ...
        )


    def forward(self, observation: Observation) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.internal_state is None:
            raise RuntimeError("Internal state not initialized.")
        
        # preprocess

        # feed

        # postprocess


        return 
