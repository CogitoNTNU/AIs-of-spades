from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from pokerenv.observation import Observation

from nn.poker_net import PokerNet
from ._cards_encoder import CardsEncoder
from ._bets_transformer import BetsTransformer
from ._state_mlp import StateMLP
from .preprocess import PreprocessedObservation, preprocess_observation


class MiheerNet(PokerNet):
    """
    Hybrid Poker network combining:
    - CNN (CardsEncoder) for card grid encoding.
    - Transformer (BetsTransformer) for betting history.
    - MLP (StateMLP) for internal hand/game states.
    - Fusion trunk with policy/value-style heads (policy logits + bet mean/std).
    """

    def __init__(
        self,
        *,
        # Cards encoder
        cards_channels: tuple[int, ...] = (32, 64, 96),
        cards_out_dim: int = 128,
        cards_dropout: float = 0.05,
        # Bets transformer
        bet_seq_len: int = 64,
        bet_feature_dim: int = 8,
        bet_model_dim: int = 128,
        bet_heads: int = 4,
        bet_layers: int = 3,
        bet_ff_dim: int = 256,
        bet_dropout: float = 0.1,
        bet_pooling: Literal["cls", "mean", "max"] = "cls",
        # State MLP
        hand_state_dim: int = 32,
        game_state_dim: int = 32,
        state_hidden_dim: int = 128,
        state_out_dim: int = 96,
        state_dropout: float = 0.1,
        # Fusion trunk
        trunk_hidden_dim: int = 256,
        trunk_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Branches
        self.cards_encoder = CardsEncoder(
            channels=cards_channels,
            dropout=cards_dropout,
            out_dim=cards_out_dim,
        )
        self.bets_encoder = BetsTransformer(
            input_dim=bet_feature_dim,
            model_dim=bet_model_dim,
            num_heads=bet_heads,
            num_layers=bet_layers,
            ff_dim=bet_ff_dim,
            dropout=bet_dropout,
            max_seq_len=bet_seq_len + 1,  # +1 for CLS token
            use_cls_token=True,
            pooling=bet_pooling,
            flatten_seq_len=bet_seq_len,
            prenorm=True,
        )
        self.state_encoder = StateMLP(
            hand_dim=hand_state_dim,
            game_dim=game_state_dim,
            hidden_dim=state_hidden_dim,
            out_dim=state_out_dim,
            dropout=state_dropout,
            use_layernorm=True,
            use_residual=False,
        )

        fusion_dim = cards_out_dim + bet_model_dim + state_out_dim
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, trunk_hidden_dim),
            nn.GELU(),
            nn.Dropout(trunk_dropout),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.GELU(),
        )

        # Heads
        self.policy_head = nn.Linear(trunk_hidden_dim, 3)  # fold/call/raise logits
        self.bet_mean_head = nn.Linear(trunk_hidden_dim, 1)
        self.bet_logvar_head = nn.Linear(trunk_hidden_dim, 1)
        self.hand_state_head = nn.Linear(trunk_hidden_dim, hand_state_dim)
        self.game_state_head = nn.Linear(trunk_hidden_dim, game_state_dim)

        # Internal state (for recurrent-style feedback)
        self._hand_state: Optional[torch.Tensor] = None
        self._game_state: Optional[torch.Tensor] = None
        self.hand_state_dim = hand_state_dim
        self.game_state_dim = game_state_dim

    # ------------------------------------------------------------------ #
    # Internal state management
    # ------------------------------------------------------------------ #
    def initialize_internal_state(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)
        self._game_state = torch.zeros(batch_size, self.game_state_dim, device=device)

    def new_hand(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def preprocess(self, observation: Observation) -> PreprocessedObservation:
        return preprocess_observation(observation)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def _encode_from_tensors(
        self,
        cards: torch.Tensor,
        bets_flat: torch.Tensor,
        hand_state: torch.Tensor,
        game_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        card_emb = self.cards_encoder(cards)
        bets_emb = self.bets_encoder(bets_flat)
        state_emb = self.state_encoder(hand_state, game_state)

        x = torch.cat([card_emb, bets_emb, state_emb], dim=-1)
        x = self.trunk(x)

        action_logits = self.policy_head(x)
        bet_mean = torch.sigmoid(self.bet_mean_head(x))
        bet_logvar = self.bet_logvar_head(x)
        bet_std = torch.exp(0.5 * bet_logvar).clamp(min=1e-4, max=1.0)

        return action_logits, bet_mean, bet_std, x

    def forward(
        self,
        observation: Observation,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._hand_state is None or self._game_state is None:
            raise RuntimeError("Internal state not initialized. Call initialize_internal_state().")

        device = next(self.parameters()).device
        pre: PreprocessedObservation = preprocess_observation(
            observation,
            device=device,
            return_padding_mask=False,
        )

        # use replay-provided states if present
        hand_state = (
            pre.hand_state.to(device)
            if pre.hand_state is not None
            else self._hand_state.to(device)
        )
        game_state = (
            pre.game_state.to(device)
            if pre.game_state is not None
            else self._game_state.to(device)
        )

        action_logits, bet_mean, bet_std, x = self._encode_from_tensors(
            cards=pre.cards.to(device),
            bets_flat=pre.bets_flat.to(device),
            hand_state=hand_state,
            game_state=game_state,
        )

        # Update internal states (detached to prevent gradient accumulation)
        self._hand_state = torch.tanh(self.hand_state_head(x)).detach()
        self._game_state = torch.tanh(self.game_state_head(x)).detach()

        # Attach internal state back to observation for replay (if supported)
        if hasattr(observation, "add_network_internal_state"):
            observation.add_network_internal_state(
                {"hand": self._hand_state, "game": self._game_state}
            )

        return action_logits, bet_mean, bet_std

    def forward_batch(
        self,
        trajectory: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        ps: list[PreprocessedObservation] = [p for p, _ in trajectory]
        n = len(ps)

        cards = torch.cat([p.cards for p in ps], dim=0).to(device)
        bets_flat = torch.cat([p.bets_flat for p in ps], dim=0).to(device)

        hand_states = torch.zeros((n, self.hand_state_dim), dtype=torch.float32, device=device)
        game_states = torch.zeros((n, self.game_state_dim), dtype=torch.float32, device=device)

        for i, p in enumerate(ps):
            if p.hand_state is not None:
                hand_states[i] = p.hand_state.to(device).reshape(self.hand_state_dim)
            if p.game_state is not None:
                game_states[i] = p.game_state.to(device).reshape(self.game_state_dim)

        action_logits, bet_mean, bet_std, _ = self._encode_from_tensors(
            cards=cards,
            bets_flat=bets_flat,
            hand_state=hand_states.detach(),
            game_state=game_states.detach(),
        )
        return action_logits, bet_mean, bet_std


__all__ = ["MiheerNet"]
