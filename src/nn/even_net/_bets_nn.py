import torch
import torch.nn as nn


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
