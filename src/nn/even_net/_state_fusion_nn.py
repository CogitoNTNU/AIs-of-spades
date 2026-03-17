import torch
import torch.nn as nn

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
