import torch
import torch.nn as nn


class CardsEncoder(nn.Module):
    """
    Encodes a set of cards (hole + board) via learned embeddings and
    a small self-attention layer that captures relational structure
    (flush draws, straight draws, pairs, ...).

    Each card is represented as:
        embed(suit) + embed(rank)   →  d_card
    A padding mask ensures unseen board cards do not contribute.

    Input
    -----
    hole_cards  : [B, 2, 2]   – each card is (suit_idx, rank_idx)
    board_cards : [B, 5, 2]   – same format; zero-padded for unseen cards
    board_mask  : [B, 5]      – True where card is present

    Output
    ------
    [B, out_dim]
    """

    def __init__(
        self,
        d_card: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        out_dim: int = 64,
    ) -> None:
        super().__init__()

        self.suit_emb = nn.Embedding(4, d_card // 2)
        self.rank_emb = nn.Embedding(13, d_card - d_card // 2)

        # Learnable token that distinguishes hole cards from board cards
        self.hole_token = nn.Parameter(torch.zeros(d_card))
        self.board_token = nn.Parameter(torch.zeros(d_card))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_card,
            nhead=n_heads,
            dim_feedforward=d_card * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.proj = nn.Linear(d_card, out_dim)
        self.d_card = d_card

    # ------------------------------------------------------------------

    def _embed_card(self, cards: torch.Tensor) -> torch.Tensor:
        """
        cards: [..., 2]  (suit_idx, rank_idx) — integer indices
        returns: [..., d_card]
        """
        suit = self.suit_emb(cards[..., 0].long())  # [..., d_card//2]
        rank = self.rank_emb(cards[..., 1].long())  # [..., d_card - d_card//2]
        return torch.cat([suit, rank], dim=-1)

    def forward(
        self,
        hole_cards: torch.Tensor,  # [B, 2, 2]
        board_cards: torch.Tensor,  # [B, 5, 2]
        board_mask: torch.Tensor,  # [B, 5]  bool, True = present
    ) -> torch.Tensor:

        B = hole_cards.size(0)

        # Embed
        hole_emb = self._embed_card(hole_cards)  # [B, 2, d_card]
        board_emb = self._embed_card(board_cards)  # [B, 5, d_card]

        # Add role tokens
        hole_emb = hole_emb + self.hole_token
        board_emb = board_emb + self.board_token

        # Concatenate → [B, 7, d_card]
        tokens = torch.cat([hole_emb, board_emb], dim=1)

        # Build key-padding mask: True = ignore
        # Hole cards are always present → False (attend)
        hole_present = torch.zeros(B, 2, dtype=torch.bool, device=tokens.device)
        board_absent = ~board_mask  # [B, 5]
        pad_mask = torch.cat([hole_present, board_absent], dim=1)  # [B, 7]

        # Self-attention over the 7 card tokens
        out = self.attn(tokens, src_key_padding_mask=pad_mask)  # [B, 7, d_card]

        # Mean-pool only over present cards
        present = ~pad_mask  # [B, 7]
        out = (out * present.unsqueeze(-1)).sum(1) / present.sum(1, keepdim=True).clamp(
            min=1
        )

        return self.proj(out)  # [B, out_dim]
