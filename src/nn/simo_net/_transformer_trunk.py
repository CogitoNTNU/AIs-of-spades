import torch
import torch.nn as nn


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
    d_model   : dimension shared by all tokens
    n_heads   : attention heads (must divide d_model)
    n_layers  : number of TransformerEncoder layers
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

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # One linear projection per non-opponent stream to map encoder
        # outputs to d_model (in case encoder out_dims differ)
        # In our design all encoders output d_model directly, so these
        # are identity-equivalent — kept for flexibility.
        self.proj_cards = nn.Linear(d_model, d_model)
        self.proj_bets = nn.Linear(d_model, d_model)
        self.proj_obs = nn.Linear(d_model, d_model)
        self.proj_hand_state = nn.Linear(d_model, d_model)
        self.proj_game_state = nn.Linear(d_model, d_model)
        self.proj_opponent = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------------

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

        # Project fixed streams → [B, 1, d_model] each
        t_cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        t_cards = self.proj_cards(cards).unsqueeze(1)
        t_bets = self.proj_bets(bets).unsqueeze(1)
        t_obs = self.proj_obs(obs).unsqueeze(1)
        t_hand = self.proj_hand_state(hand_state).unsqueeze(1)
        t_game = self.proj_game_state(game_state).unsqueeze(1)

        # Project opponent tokens → [B, n_opponents, d_model]
        t_opp = self.proj_opponent(opponents)

        # Stack all tokens → [B, 6 + n_opponents, d_model]
        tokens = torch.cat(
            [t_cls, t_cards, t_bets, t_obs, t_hand, t_game, t_opp], dim=1
        )

        # Build key-padding mask (True = ignore)
        # Fixed tokens are always attended; absent opponents are masked
        fixed_present = torch.zeros(B, 6, dtype=torch.bool, device=cards.device)
        opp_absent = ~opp_mask  # [B, n_opponents]
        pad_mask = torch.cat([fixed_present, opp_absent], dim=1)  # [B, 6+N]

        # Self-attention
        out = self.transformer(
            tokens, src_key_padding_mask=pad_mask
        )  # [B, 6+N, d_model]

        # Return CLS (index 0)
        return out[:, 0, :]  # [B, d_model]
