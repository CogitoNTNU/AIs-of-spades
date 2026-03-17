import torch
import torch.nn as nn


class BetsNN(nn.Module):
    """
    Neural network for processing betting information.
    """

    net: nn.Sequential

    def __init__(
        self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 128
    ) -> None:
        """
        Initializes the BetsNN.

        Args:
            in_dim (int): Dimension of the input betting vector.
            hidden_dim (int): Base dimension of the hidden layers.
            out_dim (int): Dimension of the output vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
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
            # Block 4
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            # Block 5 (refinement)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            # Output layer
            nn.Linear(hidden_dim // 2, out_dim),
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
