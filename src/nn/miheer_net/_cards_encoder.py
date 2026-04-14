import torch
import torch.nn as nn
from typing import Callable, List, Sequence


class ConvBlock(nn.Module):
    """
    Small configurable conv block with Conv2d → Norm(optional) → Activation → Dropout(optional).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        padding: int | None = None,
        stride: int = 1,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm: Callable[[int], nn.Module] | None = nn.BatchNorm2d,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=norm is None,
            )
        ]

        if norm is not None:
            layers.append(norm(out_channels))

        layers.append(activation())

        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass."""
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Optional residual wrapper around a ConvBlock stack.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.activation = nn.ReLU()

        conv_seq = getattr(block, "net", None)
        main_conv = None
        if isinstance(conv_seq, nn.Sequential) and len(conv_seq) > 0 and isinstance(conv_seq[0], nn.Conv2d):
            main_conv = conv_seq[0]

        if isinstance(main_conv, nn.Conv2d):
            in_c, out_c = main_conv.in_channels, main_conv.out_channels
            stride = main_conv.stride
            if in_c != out_c or stride != (1, 1):
                self.proj = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
            else:
                self.proj = nn.Identity()
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward with residual connection."""
        residual = self.proj(x)
        out = self.block(x)
        out = out + residual
        return self.activation(out)


class CardsEncoder(nn.Module):
    """
    CNN encoder for card tensors.

    Expected input shape: [B, 4, 4, 13] (batch, slots, suits, ranks).
    Internally the tensor is rearranged to [B, C, H, W] for convolution.

    Design goals:
    - Modular: configurable channel widths, depth, norm, activation, and residual usage.
    - Extensible: pluggable pooling and projection heads.
    - Stable: uses normalization and optional dropout to regularize.
    """

    def __init__(
        self,
        *,
        channels: Sequence[int] = (16, 32, 64),
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm: Callable[[int], nn.Module] | None = nn.BatchNorm2d,
        dropout: float | None = 0.05,
        use_residual: bool = True,
        global_pool: Callable[[], nn.Module] = lambda: nn.AdaptiveAvgPool2d((1, 1)),
        out_dim: int = 128,
    ) -> None:
        """
        Args:
            channels: Sequence of channel sizes for each convolutional stage.
            activation: Activation constructor (e.g., nn.ReLU, nn.GELU).
            norm: Normalization constructor (e.g., nn.BatchNorm2d, nn.GroupNorm). Set to None to disable.
            dropout: Spatial dropout probability applied after activation (per block). Set to None or 0.0 to disable.
            use_residual: If True, wrap each block with a residual connection (with channel projection if needed).
            global_pool: Factory for spatial pooling module producing [B, C, 1, 1].
            out_dim: Output embedding dimension after projection.
        """
        super().__init__()

        if len(channels) < 1:
            raise ValueError("channels must contain at least one stage.")

        blocks: List[nn.Module] = []
        in_ch = 4  # suits as channels after permute
        for c in channels:
            block = ConvBlock(
                in_channels=in_ch,
                out_channels=c,
                kernel_size=3,
                padding=1,
                activation=activation,
                norm=norm,
                dropout=dropout,
            )
            if use_residual:
                block = ResidualBlock(block)
            blocks.append(block)
            in_ch = c

        self.stem = nn.Sequential(*blocks)
        self.pool = global_pool()
        self.proj = nn.Linear(channels[-1], out_dim)

        self._out_dim = out_dim

    @property
    def out_dim(self) -> int:
        """Return embedding dimension for downstream fusion components."""
        return self._out_dim

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cards: Tensor of shape [B, 4, 4, 13] or [4, 4, 13] (auto-batched).

        Returns:
            Embedding tensor of shape [B, out_dim].
        """
        if cards.dim() == 3:
            cards = cards.unsqueeze(0)

        if cards.shape[1:] != (4, 13) and cards.shape[1:] != (4, 4, 13):
            raise ValueError(
                f"Expected cards shape [B,4,4,13] or [4,4,13], got {tuple(cards.shape)}"
            )

        # Ensure shape [B, 4, 4, 13]
        if cards.dim() == 4 and cards.shape[1:] == (4, 13):
            # likely missing slot dimension; treat batch dimension as slots -> not supported
            raise ValueError(
                "Cards tensor appears to be [B,4,13]; expected slot dimension present."
            )

        # Rearrange to [B, C, H, W]: suits → channels, slots → height, ranks → width
        x = cards.permute(0, 2, 1, 3).contiguous()  # [B, 4, 4, 13] -> [B, 4, 4, 13]

        x = self.stem(x)
        x = self.pool(x)  # [B, C, 1, 1]
        x = x.flatten(1)
        x = self.proj(x)
        return x
