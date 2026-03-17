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
        hidden_dim: int = 128,
        out_dim: int = 128,
    ) -> None:
        """
        Initializes the StateFusionNN.

        Args:
            hand_in_dim (int): Dimension of the input hand state.
            game_in_dim (int): Dimension of the input game state.
            hidden_dim (int): Base dimension of the hidden layers.
            out_dim (int): Dimension of the output vector.
        """
        super().__init__()

        in_dim = hand_in_dim + game_in_dim

        self.net = nn.Sequential(
            # Block 1 (expansion)
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # Block 2
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            # Block 3
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            # Block 4 (compression)
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            # Block 5
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            # Output
            nn.Linear(hidden_dim // 2, out_dim),
            nn.ReLU(),
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
