import torch
import torch.nn as nn


class BetsNN(nn.Module):
    """
    Encodes the hand-log history into a fixed-size vector using a Transformer.

    The hand_log is a [T, 8] matrix (T=64 actions, 8 features each).
    Each row is one action — projected to d_model and processed via
    self-attention so the encoder can learn temporal/sequential patterns
    (e.g. re-raise after raise, check-raise, etc.).

    Padding: rows initialized to -1.0 in table.py are masked out so they
    don't pollute attention. A learnable CLS token aggregates the sequence.

    Input  : [B, 64, 8]   (NON flattato — vedi nota sotto)
    Output : [B, out_dim]

    NOTA: devi aggiornare preprocess_observation in _utils.py per passare
    il hand_log come [64, 8] invece di flattatlo a [512].
    Cambia:
        bets = torch.from_numpy(observation.hand_log).float().flatten()
    in:
        bets = torch.from_numpy(observation.hand_log).float()  # [64, 8]
    E aggiorna PreprocessedObs.bets type hint: [64, 8] float.

    In SimoNet.forward() e forward_batch() il .unsqueeze(0) rimane invariato,
    ma bets_in_dim non serve più — sostituito da action_feat_dim=8.
    """

    def __init__(
        self,
        seq_len: int = 64,
        action_feat_dim: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        out_dim: int = 64,
        pad_value: float = -1.0,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.pad_value = pad_value
        self.d_model = d_model

        # Project each action row to d_model
        self.input_proj = nn.Linear(action_feat_dim, d_model)

        # Learnable CLS token — aggregates the whole sequence
        self.cls_token = nn.Parameter(torch.zeros(d_model))

        # Learnable positional encoding
        # seq_len + 1 to account for the CLS token prepended
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, bets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bets: [B, seq_len, 8]  — hand_log NON flattato
        Returns:
            [B, out_dim]
        """
        B, T, _ = bets.shape
        device = bets.device

        # Padding mask: rows where ALL features == pad_value are absent
        # shape: [B, T]
        pad_mask = (bets == self.pad_value).all(dim=-1)  # True = da ignorare

        # Project action features → [B, T, d_model]
        tokens = self.input_proj(bets)

        # Prepend CLS token → [B, T+1, d_model]
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Positional encoding
        positions = torch.arange(T + 1, device=device).unsqueeze(0)  # [1, T+1]
        tokens = tokens + self.pos_emb(positions)

        # Extend the mask with False for the CLS token (always present)
        cls_present = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_present, pad_mask], dim=1)  # [B, T+1]

        # Self-attention
        out = self.transformer(
            tokens, src_key_padding_mask=full_mask
        )  # [B, T+1, d_model]

        # CLS hidden state as global representation
        cls_out = out[:, 0, :]  # [B, d_model]

        return self.out_proj(cls_out)  # [B, out_dim]
