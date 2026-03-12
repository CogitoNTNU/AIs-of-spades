import torch
import torch.nn as nn

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
