import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerTrunk(nn.Module):
    """
    Fuses N input streams via self-attention.

    Each stream arrives as a [B, d_model] vector and is treated as one
    token.  A learnable CLS token is prepended; its final hidden state
    is used as the global representation fed to the output heads.

    Token layout (indices into the sequence):
        0  : CLS
        1  : cards
        2  : bets (hand log)
        3  : obs scalars
        4  : hand_state  (resets each hand)
        5  : game_state  (persists across hands)
        6… : opponent tokens (shuffled, zero-padded for absent seats)

    Parameters
    ----------
    d_model    : dimension shared by all tokens
    n_heads    : attention heads (must divide d_model)
    n_layers   : number of TransformerEncoder layers
    n_opponents: maximum number of opponent slots (default 5)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_opponents: int = 5,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_opponents = n_opponents

        # CLS token — init normal so it starts distinguishable from zero
        self.cls_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Per-stream LayerNorm applied before concatenation so all tokens
        # enter the first attention layer on the same scale regardless of
        # which encoder produced them.
        self.norm_cards = nn.LayerNorm(d_model)
        self.norm_bets = nn.LayerNorm(d_model)
        self.norm_obs = nn.LayerNorm(d_model)
        self.norm_hand_state = nn.LayerNorm(d_model)
        self.norm_game_state = nn.LayerNorm(d_model)
        self.norm_opponent = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable gradients than Post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        cards: torch.Tensor,  # [B, d_model]
        bets: torch.Tensor,  # [B, d_model]
        obs: torch.Tensor,  # [B, d_model]
        hand_state: torch.Tensor,  # [B, d_model]
        game_state: torch.Tensor,  # [B, d_model]
        opponents: torch.Tensor,  # [B, n_opponents, d_model]
        opp_mask: torch.Tensor,  # [B, n_opponents] bool, True = present
    ) -> torch.Tensor:
        """
        Returns
        -------
        [B, d_model]  — CLS hidden state, ready for output heads.
        """
        B = cards.size(0)

        # Normalise each stream independently before fusing.
        # This decouples the scale of each encoder from the trunk,
        # so ObsNN, BetsNN, CardsEncoder can have different activation
        # magnitudes without one stream dominating attention.
        t_cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        t_cards = self.norm_cards(cards).unsqueeze(1)
        t_bets = self.norm_bets(bets).unsqueeze(1)
        t_obs = self.norm_obs(obs).unsqueeze(1)
        t_hand = self.norm_hand_state(hand_state).unsqueeze(1)
        t_game = self.norm_game_state(game_state).unsqueeze(1)
        t_opp = self.norm_opponent(opponents)  # [B, n_opp, d_model]

        # Stack all tokens → [B, 6 + n_opponents, d_model]
        tokens = torch.cat(
            [t_cls, t_cards, t_bets, t_obs, t_hand, t_game, t_opp], dim=1
        )

        # Build key-padding mask (True = ignore).
        # Fixed tokens are always attended; absent opponent seats are masked.
        fixed_present = torch.zeros(B, 6, dtype=torch.bool, device=cards.device)
        opp_absent = ~opp_mask  # [B, n_opp]
        pad_mask = torch.cat([fixed_present, opp_absent], dim=1)  # [B, 6+N]

        out = self.transformer(
            tokens, src_key_padding_mask=pad_mask
        )  # [B, 6+N, d_model]

        return out[:, 0, :]  # CLS hidden state → [B, d_model]
