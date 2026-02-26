from typing import Literal, Tuple, Union
import torch
import torch.nn as nn


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


class PokerNet(nn.Module):
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
    value_head: nn.Linear
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
        self.value_head = nn.Linear(trunk_hidden_dim, 1)  # scalar value / score
        self.hand_state_head = nn.Linear(
            trunk_hidden_dim, hand_state_dim
        )  # for feedback loop
        self.game_state_head = nn.Linear(
            trunk_hidden_dim, game_state_dim
        )  # for feedback loop

    def forward(
        self,
        cards: torch.Tensor,
        bets: torch.Tensor,
        hand_state: torch.Tensor,
        game_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PokerNet.

        Args:
            cards (torch.Tensor): [B, 4, 4, 13] Input card tensor.
            bets (torch.Tensor): [B, bets_in_dim] Input betting tensor.
            hand_state (torch.Tensor): [B, hand_state_dim] Input hand state tensor.
            game_state (torch.Tensor): [B, game_state_dim] Input game state tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - action_logits: [B, 3] Unnormalized logits for actions (fold, call, raise).
                - value: [B, 1] Estimated value of the current state.
                - next_hand_state: [B, hand_state_dim] Updated hand state for the next step.
                - next_game_state: [B, game_state_dim] Updated game state for the next step.
        """
        fa = self.cards_branch(cards)
        fb = self.bets_branch(bets)
        fs = self.state_branch(hand_state, game_state)

        x = torch.cat([fa, fb, fs], dim=1)
        x = self.trunk(x)

        action_logits = self.policy_head(x)  # softmax outside
        value = self.value_head(x)
        next_hand_state = torch.tanh(self.hand_state_head(x))
        next_game_state = torch.tanh(self.game_state_head(x))

        return action_logits, value, next_hand_state, next_game_state
