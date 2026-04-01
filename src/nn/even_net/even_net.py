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

    Structure:
    - cards_branch (CardsCNN): Processes card information.
    - bets_branch (BetsNN): Processes betting information.
    - state_branch (StateFusionNN or StateFusionBranchedNN): Fuses hand and game states.
    - trunk: Fuses outputs from all branches.
    - Heads: Predicts action logits, value, and next states.
    """

    cards_branch: CardsCNN
    state_branch: Union[StateFusionNN, StateFusionBranchedNN]
    bets_branch: BetsNN
    trunk: nn.Sequential
    policy_head: nn.Linear
    bet_mean_head: nn.Linear
    bet_logvar_head: nn.Linear
    hand_state_head: nn.Linear
    game_state_head: nn.Linear

    def __init__(
        self,
        # BetsNN params
        bets_in_dim: int = 64 * 8,
        bets_hidden_dim: int = 64,
        bets_out_dim: int = 32,
        # CardsCNN params
        cards_out_dim: int = 64,
        # StateNN params
        hand_state_dim: int = 32,
        game_state_dim: int = 32,
        state_hidden_dim: int = 64,
        state_out_dim: int = 32,
        # Fusion trunk params
        trunk_hidden_dim: int = 128,
        *,
        state_mode: Literal["simple", "branched"] = "simple",
    ) -> None:
        """
        Initializes the PokerNet.

        Args:
            bets_in_dim (int): Input dimension for BetsNN.
            bets_hidden_dim (int): Hidden dimension for BetsNN.
            bets_out_dim (int): Output dimension for BetsNN.
            cards_out_dim (int): Output dimension for CardsCNN.
            hand_state_dim (int): Dimension for hand state.
            game_state_dim (int): Dimension for game state.
            state_hidden_dim (int): Hidden dimension for StateNN.
            state_out_dim (int): Output dimension for StateNN.
            trunk_hidden_dim (int): Hidden dimension for the fusion trunk.
            state_mode (Literal["simple", "branched"]): Mode for state fusion.
        """
        super().__init__()

        if state_mode == "simple":
            state_nn = StateFusionNN
        elif state_mode == "branched":
            state_nn = StateFusionBranchedNN
        else:
            raise ValueError(
                f"Invalid state_mode={state_mode!r}. Use 'simple' or 'branched'."
            )

        self.cards_branch = CardsCNN(
            cards_out_dim,
        )
        self.state_branch = state_nn(
            hand_state_dim,
            game_state_dim,
            state_hidden_dim,
            state_out_dim,
        )
        self.bets_branch = BetsNN(
            bets_in_dim,
            bets_hidden_dim,
            bets_out_dim,
        )

        # Fusion trunk
        self.trunk = nn.Sequential(
            nn.Linear(
                cards_out_dim + bets_out_dim + state_out_dim,
                trunk_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.ReLU(),
        )

        # Heads
        self.policy_head = nn.Linear(trunk_hidden_dim, 3)  # fold/call/raise logits
        self.bet_mean_head = nn.Linear(trunk_hidden_dim, 1)
        self.bet_logvar_head = nn.Linear(trunk_hidden_dim, 1)
        self.hand_state_head = nn.Linear(
            trunk_hidden_dim, hand_state_dim
        )  # for feedback loop
        self.game_state_head = nn.Linear(
            trunk_hidden_dim, game_state_dim
        )  # for feedback loop

        self._hand_state = None
        self._game_state = None

        self.hand_state_dim = hand_state_dim
        self.game_state_dim = game_state_dim

    def initialize_internal_state(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)
        self._game_state = torch.zeros(batch_size, self.game_state_dim, device=device)

    def new_hand(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)

    def forward(
        self,
        observation: Observation,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self._hand_state is None or self._game_state is None:
            raise RuntimeError("Internal state not initialized.")

        # extract cards and bets from the observations
        cards, bets, network_internal_state_hand, network_internal_state_game = (
            preprocess_observation(observation)
        )

        # ensure batch dimension
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

        # branches
        fa = self.cards_branch(cards)
        fb = self.bets_branch(bets)
        fs = self.state_branch(hand_state, game_state)

        # fusion
        x = torch.cat([fa, fb, fs], dim=1)
        x = self.trunk(x)

        action_logits = self.policy_head(x)

        bet_mean = torch.sigmoid(self.bet_mean_head(x))
        bet_logvar = self.bet_logvar_head(x)
        bet_std = torch.exp(0.5 * bet_logvar).clamp(min=1e-4, max=1.0)

        # update internal state
        self._hand_state = torch.tanh(self.hand_state_head(x)).detach()
        self._game_state = torch.tanh(self.game_state_head(x)).detach()

        return action_logits, bet_mean, bet_std
