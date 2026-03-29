import torch
import torch.nn as nn


class ObsNN(nn.Module):
    """
    Encodes the scalar features extracted from the current observation:
    street, pot, bet_to_match, minimum_raise, player position/stack/
    money_in_pot/bet_this_street, bet_range bounds, and per-opponent
    features (position, state, stack, money_in_pot, bet_this_street,
    is_all_in).

    The input vector is assembled by preprocess_observation and has a
    fixed size regardless of how many opponents are at the table
    (missing seats are zero-padded).

    Input  : [B, in_dim]
    Output : [B, out_dim]
    """

    net: nn.Sequential

    def __init__(
        self,
        in_dim: int = 48,  # 8 self-features + 5×8 opponent-features
        hidden_dim: int = 64,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, in_dim]
        Returns:
            [B, out_dim]
        """
        return self.net(obs)
