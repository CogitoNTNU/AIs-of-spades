import torch
import torch.nn as nn


class BetsNN(nn.Module):
    """
    Encodes the hand-log history into a fixed-size vector.

    The hand_log is a [T, 8] matrix (T actions, 8 features each).
    We flatten it and pass it through an MLP.

    Input  : [B, T*8]   (pre-flattened in preprocess_observation)
    Output : [B, out_dim]
    """

    net: nn.Sequential

    def __init__(
        self,
        in_dim: int = 512,  # 64 rows × 8 cols
        hidden_dim: int = 128,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, bets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bets: [B, in_dim]
        Returns:
            [B, out_dim]
        """
        return self.net(bets)
