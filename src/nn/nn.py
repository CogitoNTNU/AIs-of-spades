from typing import Literal, Tuple, Union
from nn.poker_net import PokerNet
import torch
import torch.nn as nn
from pokerenv.observation import Observation


class CardsCNN(nn.Module):
    """
    CNN for processing card information.
    Expects input of shape [B, 4, 4, 13] representing cards in a 2D grid.
    """

    net: nn.Sequential
    proj: nn.Linear

    def __init__(self, out_dim: int = 128) -> None:
        """
        Initializes the CardsCNN.

        Args:
            out_dim (int): The dimension of the output vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=4, out_channels=16, kernel_size=3, padding=1
            ),  # [B,16,4,13]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B,32,4,13]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B,32,1,1]
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CardsCNN.

        Args:
            x (torch.Tensor): [B, 4, 4, 13] Input tensor.

        Returns:
            torch.Tensor: [B, out_dim] Output tensor.
        """
        x = self.net(x)
        x = x.flatten(1)  # [B,32]
        x = self.proj(x)  # [B,out_dim]
        return x


class StateFusionBranchedNN(nn.Module):
    """
    Neural network for fusing hand and game states using separate branches.
    """

    hand_branch: nn.Sequential
    game_branch: nn.Sequential
    net: nn.Sequential

    def __init__(
        self,
        hand_in_dim: int,
        game_in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
    ) -> None:
        """
        Initializes the StateFusionBranchedNN.

        Args:
            hand_in_dim (int): Dimension of the input hand state.
            game_in_dim (int): Dimension of the input game state.
            hidden_dim (int): Dimension of the hidden layers.
            out_dim (int): Dimension of the output vector.
        """
        super().__init__()
        self.hand_branch = nn.Sequential(
            nn.Linear(hand_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.game_branch = nn.Sequential(
            nn.Linear(game_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hand: torch.Tensor, game: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StateFusionBranchedNN.

        Args:
            hand (torch.Tensor): [B, hand_in_dim] Input hand state tensor.
            game (torch.Tensor): [B, game_in_dim] Input game state tensor.

        Returns:
            torch.Tensor: [B, out_dim] Fused state tensor.
        """
        fh = self.hand_branch(hand)
        fg = self.game_branch(game)
        x = torch.cat([fh, fg], dim=1)
        return self.net(x)


class StateFusionNN(nn.Module):
    """
    Neural network for fusing hand and game states by concatenating them.
    """

    net: nn.Sequential

    def __init__(
        self,
        hand_in_dim: int,
        game_in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
    ) -> None:
        """
        Initializes the StateFusionNN.

        Args:
            hand_in_dim (int): Dimension of the input hand state.
            game_in_dim (int): Dimension of the input game state.
            hidden_dim (int): Dimension of the hidden layers.
            out_dim (int): Dimension of the output vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hand_in_dim + game_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hand: torch.Tensor, game: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StateFusionNN.

        Args:
            hand (torch.Tensor): [B, hand_in_dim] Input hand state tensor.
            game (torch.Tensor): [B, game_in_dim] Input game state tensor.

        Returns:
            torch.Tensor: [B, out_dim] Fused state tensor.
        """
        x = torch.cat([hand, game], dim=1)
        return self.net(x)


class BetsNN(nn.Module):
    """
    Neural network for processing betting information.
    """

    net: nn.Sequential

    def __init__(
        self, in_dim: int = 128, hidden_dim: int = 64, out_dim: int = 32
    ) -> None:
        """
        Initializes the BetsNN.

        Args:
            in_dim (int): Dimension of the input betting vector.
            hidden_dim (int): Dimension of the hidden layers.
            out_dim (int): Dimension of the output vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, bets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BetsNN.

        Args:
            bets (torch.Tensor): [B, in_dim] Input betting tensor.

        Returns:
            torch.Tensor: [B, out_dim] Processed betting features.
        """
        return self.net(bets)


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
        bets_in_dim: int = 128,
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

    def preprocess_observation(
        self, observation: Observation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a single Observation object into card and betting tensors.

        Returns:
            cards: [4, 4, 13] tensor
            bets: [128] tensor (flattened hand log)
        """

        print(observation)
        # --------- CARDS ---------
        card_tensor = torch.zeros((4, 4, 13), dtype=torch.float32)

        # Encode hand cards
        for i, card_obs in enumerate(observation.hand_cards.cards):
            suit_idx = int(card_obs.suit) % 4
            rank_idx = int(card_obs.rank)

            # channel = suit
            # slot = card index
            card_tensor[suit_idx, i, rank_idx] = 1.0

        # Encode table cards
        offset = len(observation.hand_cards.cards)

        for j, card_obs in enumerate(observation.table_cards.cards):
            idx = j + offset
            if idx >= 4:
                break

            suit_idx = int(card_obs.suit)
            rank_idx = int(card_obs.rank)

            card_tensor[suit_idx, idx, rank_idx] = 1.0

        # --------- BET HISTORY ---------

        # convert numpy log to tensor
        bets_tensor = torch.from_numpy(observation.hand_log).float()

        # flatten [32,4] → [128]
        bets_tensor = bets_tensor.flatten()

        return card_tensor, bets_tensor

    def forward(
        self,
        observation: Observation,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self._hand_state is None or self._game_state is None:
            raise RuntimeError("Internal state not initialized.")

        # extract cards and bets from the observations
        cards, bets = self.preprocess_observation(observation)

        # ensure batch dimension
        if cards.dim() == 3:
            cards = cards.unsqueeze(0)

        if bets.dim() == 1:
            bets = bets.unsqueeze(0)

        # branches
        fa = self.cards_branch(cards)
        fb = self.bets_branch(bets)
        fs = self.state_branch(self._hand_state, self._game_state)

        # fusion
        x = torch.cat([fa, fb, fs], dim=1)
        x = self.trunk(x)

        action_logits = self.policy_head(x)

        bet_mean = self.bet_mean_head(x)
        bet_logvar = self.bet_logvar_head(x)
        bet_variance = torch.exp(bet_logvar)

        # update internal state
        self._hand_state = torch.tanh(self.hand_state_head(x))
        self._game_state = torch.tanh(self.game_state_head(x))

        return action_logits, bet_mean, bet_variance
