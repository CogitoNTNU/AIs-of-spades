from typing import Literal, Tuple, Union
from nn.poker_net import PokerNet
import torch
import torch.nn as nn
from pokerenv.observation import Observation
from ._bets_nn import BetsNN
from ._cards_cnn import CardsCNN
from ._state_fusion_branched_nn import StateFusionBranchedNN
from ._state_fusion_nn import StateFusionNN
from ._utils import preprocess_observation


class EvenNet(PokerNet):
    """
    Integrated Poker network that combines cards, bets, and game state.
    """

    cards_branch: CardsCNN
    state_branch: Union[StateFusionNN, StateFusionBranchedNN]
    bets_branch: BetsNN
    trunk: nn.Sequential

    policy_head: nn.Module
    bet_mean_head: nn.Module
    bet_logvar_head: nn.Module
    hand_state_head: nn.Linear
    game_state_head: nn.Linear

    def __init__(
        self,
        # BetsNN params (UPDATED)
        bets_in_dim: int = 128,
        bets_hidden_dim: int = 128,
        bets_out_dim: int = 128,
        # CardsCNN params (UPDATED)
        cards_out_dim: int = 256,
        # StateNN params (UPDATED)
        hand_state_dim: int = 128,
        game_state_dim: int = 128,
        state_hidden_dim: int = 128,
        state_out_dim: int = 128,
        # Fusion trunk params (UPDATED)
        trunk_hidden_dim: int = 256,
        *,
        state_mode: Literal["simple", "branched"] = "simple",
    ) -> None:
        super().__init__()

        if state_mode == "simple":
            state_nn = StateFusionNN
        elif state_mode == "branched":
            state_nn = StateFusionBranchedNN
        else:
            raise ValueError(
                f"Invalid state_mode={state_mode!r}. Use 'simple' or 'branched'."
            )

        # Branches (aligned with upgraded networks)
        self.cards_branch = CardsCNN(cards_out_dim)
        self.bets_branch = BetsNN(bets_in_dim, bets_hidden_dim, bets_out_dim)
        self.state_branch = state_nn(
            hand_state_dim,
            game_state_dim,
            state_hidden_dim,
            state_out_dim,
        )

        fusion_in_dim = cards_out_dim + bets_out_dim + state_out_dim

        # Deeper trunk
        self.trunk = nn.Sequential(
            nn.Linear(fusion_in_dim, trunk_hidden_dim),
            nn.BatchNorm1d(trunk_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim * 2),
            nn.BatchNorm1d(trunk_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.35),
            nn.Linear(trunk_hidden_dim * 2, trunk_hidden_dim),
            nn.BatchNorm1d(trunk_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.ReLU(),
        )

        # Heads (slightly deeper)
        self.policy_head = nn.Sequential(
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(trunk_hidden_dim // 2, 3),
        )

        self.bet_mean_head = nn.Sequential(
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(trunk_hidden_dim // 2, 1),
        )

        self.bet_logvar_head = nn.Sequential(
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(trunk_hidden_dim // 2, 1),
        )

        self.hand_state_head = nn.Linear(trunk_hidden_dim, hand_state_dim)
        self.game_state_head = nn.Linear(trunk_hidden_dim, game_state_dim)

        self._hand_state = None
        self._game_state = None

        self.hand_state_dim = hand_state_dim
        self.game_state_dim = game_state_dim

    def initialize_internal_state(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)
        self._game_state = torch.zeros(batch_size, self.game_state_dim, device=device)

    def reset_hand(self):
        """
        Reset only the hand-specific state, keep game state.
        """
        if self._hand_state is None:
            raise RuntimeError("Internal state not initialized.")

        self._hand_state = torch.zeros_like(self._hand_state)

    def forward(
        self,
        observation: Observation,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self._hand_state is None or self._game_state is None:
            raise RuntimeError("Internal state not initialized.")

        cards, bets, network_internal_state_hand, network_internal_state_game = (
            preprocess_observation(observation)
        )

        # Ensure batch dimension
        if cards.dim() == 3:
            cards = cards.unsqueeze(0)

        if bets.dim() == 1:
            bets = bets.unsqueeze(0)

        hand_state = (
            self._hand_state
            if network_internal_state_hand is None
            else network_internal_state_hand
        )

        game_state = (
            self._game_state
            if network_internal_state_game is None
            else network_internal_state_game
        )

        observation.add_network_internal_state({"hand": hand_state, "game": game_state})

        # Branches
        fa = self.cards_branch(cards)
        fb = self.bets_branch(bets)
        fs = self.state_branch(hand_state, game_state)

        # Fusion
        x = torch.cat([fa, fb, fs], dim=1)
        x = self.trunk(x)

        # Outputs
        action_logits = self.policy_head(x)

        bet_mean = torch.sigmoid(self.bet_mean_head(x))
        bet_logvar = self.bet_logvar_head(x)
        bet_std = torch.exp(0.5 * bet_logvar).clamp(min=1e-4, max=1.0)

        # Update internal states
        self._hand_state = torch.tanh(self.hand_state_head(x)).detach()
        self._game_state = torch.tanh(self.game_state_head(x)).detach()

        return action_logits, bet_mean, bet_std
