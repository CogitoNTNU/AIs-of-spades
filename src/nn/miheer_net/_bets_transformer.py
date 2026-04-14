import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BetsTransformer(nn.Module):
    """
    Transformer encoder for betting history.

    Design goals:
    - Handles both pre-shaped sequences [B, T, F] and flattened logs [B, F_flat] with configurable seq_len.
    - Optional CLS token with configurable pooling strategy.
    - Stable defaults (prenorm + dropout) suitable for research iterations.

    Expected inputs:
        - bets: torch.Tensor of shape [B, T, F] or [B, F_flat].
        - padding_mask (optional): Bool tensor [B, T] where True indicates PAD positions.

    Output:
        - betting_embedding: torch.Tensor of shape [B, model_dim].
    """

    def __init__(
        self,
        *,
        input_dim: int = 4,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        use_cls_token: bool = True,
        pooling: Literal["cls", "mean", "max"] = "cls",
        flatten_seq_len: Optional[int] = None,
        prenorm: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Feature dimension per time-step before projection.
            model_dim: Transformer model dimension.
            num_heads: Attention heads.
            num_layers: Number of encoder layers.
            ff_dim: Feed-forward hidden dimension.
            dropout: Dropout applied throughout.
            max_seq_len: Maximum supported sequence length for positional encoding.
            use_cls_token: Whether to prepend a learnable CLS token.
            pooling: Pooling strategy over sequence ("cls", "mean", or "max").
            flatten_seq_len: If provided, reshape flat inputs [B, F_flat] into
                             [B, flatten_seq_len, F_flat/flatten_seq_len].
            prenorm: Use pre-layernorm encoder blocks if True, else post-norm.
        """
        super().__init__()

        if flatten_seq_len is not None and flatten_seq_len <= 0:
            raise ValueError("flatten_seq_len must be positive or None.")

        self.flatten_seq_len = flatten_seq_len
        self.use_cls_token = use_cls_token
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(model_dim, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=prenorm,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        self.norm_out = nn.LayerNorm(model_dim)

        self._out_dim = model_dim

    @property
    def out_dim(self) -> int:
        """Return the embedding dimension produced by the encoder."""
        return self._out_dim

    def _reshape_if_flat(self, bets: torch.Tensor) -> torch.Tensor:
        """
        Reshape flattened input [B, F_flat] → [B, T, F] if flatten_seq_len is set.
        """
        if bets.dim() == 3:
            return bets

        if self.flatten_seq_len is None:
            raise ValueError(
                "Received flat bets tensor but flatten_seq_len is not configured."
            )

        batch, flat_dim = bets.shape
        if flat_dim % self.flatten_seq_len != 0:
            raise ValueError(
                f"Flat bets dimension {flat_dim} not divisible by flatten_seq_len={self.flatten_seq_len}."
            )
        feat_dim = flat_dim // self.flatten_seq_len
        return bets.view(batch, self.flatten_seq_len, feat_dim)

    def forward(
        self,
        bets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bets: [B, T, F] or [B, F_flat].
            padding_mask: Optional bool mask [B, T] (True for padding positions).
        Returns:
            betting_embedding: [B, model_dim]
        """
        x = self._reshape_if_flat(bets)  # [B, T, F]
        B, T, _ = x.shape

        x = self.input_proj(x)  # [B, T, D]

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, 1, -1)  # [B,1,D]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+T, D]
            if padding_mask is not None:
                padding_mask = torch.cat(
                    [torch.zeros((B, 1), dtype=padding_mask.dtype, device=padding_mask.device), padding_mask],
                    dim=1,
                )
            T = T + 1

        if T > self.pos_encoding.pe.size(1):
            raise ValueError(
                f"Sequence length {T} exceeds configured max_seq_len={self.pos_encoding.pe.size(1)}."
            )

        x = self.pos_encoding(x)

        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        if self.pooling == "cls":
            if not self.use_cls_token:
                raise ValueError("CLS pooling requires use_cls_token=True.")
            pooled = encoded[:, 0]  # [B, D]
        elif self.pooling == "mean":
            if padding_mask is None:
                pooled = encoded.mean(dim=1)
            else:
                valid = (~padding_mask).float()
                pooled = (encoded * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
                    dim=1, keepdim=True
                ).clamp(min=1e-6)
        elif self.pooling == "max":
            if padding_mask is None:
                pooled, _ = encoded.max(dim=1)
            else:
                mask = padding_mask.unsqueeze(-1).expand_as(encoded)
                masked = encoded.masked_fill(mask, float("-inf"))
                pooled, _ = masked.max(dim=1)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        return self.norm_out(pooled)
