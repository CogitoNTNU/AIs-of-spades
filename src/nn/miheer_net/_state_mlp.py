import torch
import torch.nn as nn
from typing import Callable, Optional


class MLPBlock(nn.Module):
    """
    Lightweight MLP block: Linear -> Activation -> Dropout(optional) -> LayerNorm(optional).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim), activation()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        if use_layernorm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass."""
        return self.net(x)


class StateMLP(nn.Module):
    """
    MLP encoder for internal game + hand state.

    - Two independent branches for hand and game vectors.
    - Concatenates branch outputs and projects to a shared embedding.
    - Optional residual skip on the fused projection.

    Expected inputs:
        hand: [B, hand_dim]
        game: [B, game_dim]

    Output:
        fused embedding: [B, out_dim]
    """

    def __init__(
        self,
        *,
        hand_dim: int,
        game_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.hand_branch = nn.Sequential(
            MLPBlock(hand_dim, hidden_dim, activation=activation, dropout=dropout, use_layernorm=use_layernorm),
            MLPBlock(hidden_dim, hidden_dim, activation=activation, dropout=dropout, use_layernorm=use_layernorm),
        )
        self.game_branch = nn.Sequential(
            MLPBlock(game_dim, hidden_dim, activation=activation, dropout=dropout, use_layernorm=use_layernorm),
            MLPBlock(hidden_dim, hidden_dim, activation=activation, dropout=dropout, use_layernorm=use_layernorm),
        )
        self.fuse = nn.Sequential(
            MLPBlock(2 * hidden_dim, hidden_dim, activation=activation, dropout=dropout, use_layernorm=use_layernorm),
            nn.Linear(hidden_dim, out_dim),
        )
        self.use_residual = use_residual and (out_dim == 2 * hidden_dim)
        self._out_dim = out_dim

    @property
    def out_dim(self) -> int:
        """Embedding dimension produced by the encoder."""
        return self._out_dim

    def forward(self, hand: torch.Tensor, game: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand: [B, hand_dim]
            game: [B, game_dim]
        Returns:
            fused embedding [B, out_dim]
        """
        fh = self.hand_branch(hand)
        fg = self.game_branch(game)
        fused = torch.cat([fh, fg], dim=-1)
        out = self.fuse(fused)
        if self.use_residual:
            out = out + fused
        return out
