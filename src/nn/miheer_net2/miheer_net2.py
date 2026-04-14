from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from pokerenv.observation import Observation

from nn.poker_net import PokerNet
from nn.miheer_net._cards_encoder import CardsEncoder
from nn.miheer_net._state_mlp import StateMLP

from .preprocess import (
    MAX_OPPONENTS,
    OPP_FEAT_DIM,
    OBS_SCALAR_DIM,
    PreprocessedObservation2,
    preprocess_observation,
)


class TemporalDecayBetsTransformer(nn.Module):
    """
    Betting transformer with recency-biased temporal decay before attention.
    """

    def __init__(
        self,
        *,
        input_dim: int = 8,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len + 1, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Learnable, positive decay strength. Higher => stronger recency emphasis.
        self.decay_logit = nn.Parameter(torch.tensor(0.0))

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.max_seq_len = max_seq_len

    def forward(self, bets_seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # bets_seq: [B, T, F], padding_mask: [B, T] (True = padding)
        bsz, t, _ = bets_seq.shape
        if t > self.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.max_seq_len}")

        x = self.input_proj(bets_seq)

        # Temporal decay: older actions get lower weight than recent ones.
        age = torch.arange(t, device=x.device, dtype=x.dtype)
        age = (t - 1 - age).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        decay = torch.exp(-torch.relu(self.decay_logit) * age)
        x = x * decay

        cls = self.cls_token.expand(bsz, 1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+T, D]

        pos = torch.arange(t + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

        full_mask = torch.cat(
            [torch.zeros((bsz, 1), dtype=torch.bool, device=x.device), padding_mask],
            dim=1,
        )

        h = self.encoder(x, src_key_padding_mask=full_mask)
        return self.norm(h[:, 0, :])


class StreamAttentionFusion(nn.Module):
    """
    Attention over stream tokens, returning a single fused vector.
    """

    def __init__(self, d_model: int, num_heads: int = 4, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, S, D]
        bsz = tokens.size(0)
        cls = self.cls.expand(bsz, 1, -1)
        h = self.enc(torch.cat([cls, tokens], dim=1))
        return self.norm(h[:, 0, :])


class MiheerNet2(PokerNet):
    """
    MiheerNet2 architecture:
    Cards CNN -> Bets Transformer (temporal decay) -> State Encoder -> Opponent Memory (lightweight)
    -> Dynamic Stream Gating -> Stream Attention Fusion -> Hierarchical Memory Update
    -> Confidence-Aware Policy Head -> (policy, bet mean/std, next states).
    """

    def __init__(
        self,
        *,
        cards_channels: tuple[int, ...] = (32, 64, 96),
        cards_out_dim: int = 128,
        cards_dropout: float = 0.05,
        bet_seq_len: int = 64,
        bet_feature_dim: int = 8,
        bet_model_dim: int = 128,
        bet_heads: int = 4,
        bet_layers: int = 3,
        bet_ff_dim: int = 256,
        bet_dropout: float = 0.1,
        hand_state_dim: int = 64,
        game_state_dim: int = 64,
        state_hidden_dim: int = 128,
        state_out_dim: int = 128,
        state_dropout: float = 0.1,
        opp_mem_dim: int = 64,
        fusion_dim: int = 128,
        fusion_heads: int = 4,
        fusion_layers: int = 2,
        trunk_dropout: float = 0.1,
        opp_memory_momentum: float = 0.90,
        hand_update_alpha: float = 0.65,
        game_update_alpha: float = 0.20,
    ) -> None:
        super().__init__()

        self.hand_state_dim = hand_state_dim
        self.game_state_dim = game_state_dim
        self.opp_mem_dim = opp_mem_dim
        self.opp_memory_momentum = opp_memory_momentum
        self.hand_update_alpha = hand_update_alpha
        self.game_update_alpha = game_update_alpha

        # Streams
        self.cards_encoder = CardsEncoder(
            channels=cards_channels,
            dropout=cards_dropout,
            out_dim=cards_out_dim,
        )
        self.bets_encoder = TemporalDecayBetsTransformer(
            input_dim=bet_feature_dim,
            model_dim=bet_model_dim,
            num_heads=bet_heads,
            num_layers=bet_layers,
            ff_dim=bet_ff_dim,
            dropout=bet_dropout,
            max_seq_len=bet_seq_len,
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
        self.obs_encoder = nn.Sequential(
            nn.Linear(OBS_SCALAR_DIM, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.opp_encoder = nn.Sequential(
            nn.Linear(OPP_FEAT_DIM, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.cards_proj = nn.Linear(cards_out_dim, fusion_dim)
        self.bets_proj = nn.Linear(bet_model_dim, fusion_dim)
        self.state_proj = nn.Linear(state_out_dim, fusion_dim)
        self.opp_mem_proj = nn.Linear(opp_mem_dim, fusion_dim)

        # Dynamic stream gating: one gate per stream [cards, bets, state, obs, opp_mem]
        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 5, fusion_dim),
            nn.GELU(),
            nn.Dropout(trunk_dropout),
            nn.Linear(fusion_dim, 5),
        )

        # Stream attention fusion
        self.stream_fusion = StreamAttentionFusion(
            d_model=fusion_dim,
            num_heads=fusion_heads,
            layers=fusion_layers,
            dropout=trunk_dropout,
        )

        # Hierarchical memory update heads
        self.hand_candidate_head = nn.Linear(fusion_dim, hand_state_dim)
        self.game_candidate_head = nn.Linear(fusion_dim, game_state_dim)
        self.opp_memory_head = nn.Linear(fusion_dim, opp_mem_dim)

        # Confidence-aware action and bet heads
        self.policy_head = nn.Linear(fusion_dim, 3)
        self.confidence_head = nn.Linear(fusion_dim, 1)
        self.bet_mean_head = nn.Linear(fusion_dim, 1)
        self.bet_logvar_head = nn.Linear(fusion_dim, 1)

        self._hand_state: Optional[torch.Tensor] = None
        self._game_state: Optional[torch.Tensor] = None
        self._opp_memory: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def initialize_internal_state(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)
        self._game_state = torch.zeros(batch_size, self.game_state_dim, device=device)
        self._opp_memory = torch.zeros(batch_size, self.opp_mem_dim, device=device)

    def new_hand(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)

    def preprocess(self, observation: Observation) -> PreprocessedObservation2:
        return preprocess_observation(observation)

    def _opponent_token_summary(self, opp_vecs: torch.Tensor, opp_mask: torch.Tensor) -> torch.Tensor:
        # opp_vecs: [B, 5, F], opp_mask: [B, 5]
        opp_tok = self.opp_encoder(opp_vecs)
        mask = opp_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (opp_tok * mask).sum(dim=1) / denom

    def _encode(
        self,
        *,
        cards: torch.Tensor,
        bets_seq: torch.Tensor,
        padding_mask: torch.Tensor,
        obs_scalars: torch.Tensor,
        opp_vecs: torch.Tensor,
        opp_mask: torch.Tensor,
        hand_state: torch.Tensor,
        game_state: torch.Tensor,
        opp_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cards_e = self.cards_proj(self.cards_encoder(cards))
        bets_e = self.bets_proj(self.bets_encoder(bets_seq, padding_mask))
        state_e = self.state_proj(self.state_encoder(hand_state, game_state))
        obs_e = self.obs_encoder(obs_scalars)

        opp_summary = self._opponent_token_summary(opp_vecs, opp_mask)
        opp_mem_input = self.opp_mem_proj(opp_memory) + opp_summary

        streams = torch.stack([cards_e, bets_e, state_e, obs_e, opp_mem_input], dim=1)

        # Dynamic stream gating
        g = torch.sigmoid(self.gate_mlp(torch.cat([cards_e, bets_e, state_e, obs_e, opp_mem_input], dim=-1)))
        streams = streams * g.unsqueeze(-1)

        fused = self.stream_fusion(streams)

        # Confidence-aware policy: scale logits by confidence-derived factor.
        confidence = torch.sigmoid(self.confidence_head(fused))
        policy_scale = 0.5 + confidence
        action_logits = self.policy_head(fused) * policy_scale

        bet_mean = torch.sigmoid(self.bet_mean_head(fused))
        bet_std = torch.exp(0.5 * self.bet_logvar_head(fused)).clamp(min=1e-4, max=1.0)

        hand_cand = torch.tanh(self.hand_candidate_head(fused))
        game_cand = torch.tanh(self.game_candidate_head(fused))
        opp_cand = torch.tanh(self.opp_memory_head(fused))

        return action_logits, bet_mean, bet_std, hand_cand, game_cand, opp_cand

    def forward(self, observation: Observation) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._hand_state is None or self._game_state is None or self._opp_memory is None:
            raise RuntimeError("Internal state not initialized. Call initialize_internal_state().")

        device = next(self.parameters()).device
        p = preprocess_observation(observation, device=device)

        hand_state = p.hand_state if p.hand_state is not None else self._hand_state
        game_state = p.game_state if p.game_state is not None else self._game_state
        opp_memory = p.opp_memory if p.opp_memory is not None else self._opp_memory

        (
            action_logits,
            bet_mean,
            bet_std,
            hand_cand,
            game_cand,
            opp_cand,
        ) = self._encode(
            cards=p.cards,
            bets_seq=p.bets_seq,
            padding_mask=p.padding_mask,
            obs_scalars=p.obs_scalars,
            opp_vecs=p.opp_vecs,
            opp_mask=p.opp_mask,
            hand_state=hand_state,
            game_state=game_state,
            opp_memory=opp_memory,
        )

        # Hierarchical memory update:
        # - hand state: fast update
        # - game state: slower update
        # - opponent memory: EMA update
        self._hand_state = (
            self.hand_update_alpha * hand_cand + (1.0 - self.hand_update_alpha) * hand_state
        ).detach()
        self._game_state = (
            self.game_update_alpha * game_cand + (1.0 - self.game_update_alpha) * game_state
        ).detach()
        self._opp_memory = (
            self.opp_memory_momentum * opp_memory + (1.0 - self.opp_memory_momentum) * opp_cand
        ).detach()

        if hasattr(observation, "add_network_internal_state"):
            observation.add_network_internal_state(
                {
                    "hand": self._hand_state,
                    "game": self._game_state,
                    "opp_memory": self._opp_memory,
                }
            )

        return action_logits, bet_mean, bet_std

    def forward_batch(self, trajectory: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        ps: list[PreprocessedObservation2] = [p for p, _ in trajectory]
        n = len(ps)

        cards = torch.cat([p.cards for p in ps], dim=0).to(device)
        bets_seq = torch.cat([p.bets_seq for p in ps], dim=0).to(device)
        padding_mask = torch.cat([p.padding_mask for p in ps], dim=0).to(device)
        obs_scalars = torch.cat([p.obs_scalars for p in ps], dim=0).to(device)
        opp_vecs = torch.cat([p.opp_vecs for p in ps], dim=0).to(device)
        opp_mask = torch.cat([p.opp_mask for p in ps], dim=0).to(device)

        hand_states = torch.zeros((n, self.hand_state_dim), dtype=torch.float32, device=device)
        game_states = torch.zeros((n, self.game_state_dim), dtype=torch.float32, device=device)
        opp_memories = torch.zeros((n, self.opp_mem_dim), dtype=torch.float32, device=device)

        for i, p in enumerate(ps):
            if p.hand_state is not None:
                hand_states[i] = p.hand_state.to(device).reshape(self.hand_state_dim)
            if p.game_state is not None:
                game_states[i] = p.game_state.to(device).reshape(self.game_state_dim)
            if p.opp_memory is not None:
                opp_memories[i] = p.opp_memory.to(device).reshape(self.opp_mem_dim)

        action_logits, bet_mean, bet_std, _, _, _ = self._encode(
            cards=cards,
            bets_seq=bets_seq,
            padding_mask=padding_mask,
            obs_scalars=obs_scalars,
            opp_vecs=opp_vecs,
            opp_mask=opp_mask,
            hand_state=hand_states.detach(),
            game_state=game_states.detach(),
            opp_memory=opp_memories.detach(),
        )
        return action_logits, bet_mean, bet_std


__all__ = ["MiheerNet2"]
