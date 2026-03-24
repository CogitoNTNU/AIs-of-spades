"""
simo_net.py

SimoNet — redesigned poker policy network.

Architecture overview
---------------------

Four input encoders produce d_model-dimensional tokens:

    CardsEncoder   — hole + board cards via embedding + self-attention
    BetsNN         — flattened hand_log (action history) via MLP
    ObsNN          — scalar observation features via MLP
    Linear         — hand_state  (resets each hand,  d_model-dim)
    Linear         — game_state  (persists all game,  d_model-dim)
    Linear × 5    — opponent tokens (shuffled, absent seats zeroed)

All tokens + a learnable CLS token are fused by a TransformerTrunk
(multi-head self-attention).  The CLS hidden state drives three heads:

    policy_head    → 3 logits  (fold / call / bet)
    bet_mean_head  → scalar ∈ (0,1)  (fraction of bet range)
    bet_std_head   → scalar > 0
    hand_state_out → new hand_state  (fed back next step, reset at new_hand)
    game_state_out → new game_state  (fed back next step, never reset)

Opponent shuffle
----------------
At each forward pass the opponent tokens are randomly permuted so the
network cannot rely on slot position to identify a player.  The
permutation is consistent within a call (same B-dim shuffle applied).
The opp_mask is permuted identically.
"""

import random
import torch
import torch.nn as nn
from typing import Tuple

from nn.poker_net import PokerNet
from pokerenv.observation import Observation

from ._cards_encoder import CardsEncoder
from ._bets_nn import BetsNN
from ._obs_nn import ObsNN
from ._transformer_trunk import TransformerTrunk
from ._utils import preprocess_observation, MAX_OPPONENTS, OPP_FEAT_DIM


class SimoNet(PokerNet):
    """
    Parameters
    ----------
    d_model          : shared token dimension throughout the network
    cards_heads      : attention heads inside CardsEncoder
    cards_layers     : transformer layers inside CardsEncoder
    trunk_heads      : attention heads inside TransformerTrunk
    trunk_layers     : transformer layers inside TransformerTrunk
    bets_hidden      : hidden dim of BetsNN
    bets_in_dim      : input dim of BetsNN  (hand_log rows × cols)
    obs_in_dim       : input dim of ObsNN   (scalar obs features)
    obs_hidden       : hidden dim of ObsNN
    hand_state_dim   : dimension of the per-hand recurrent state
    game_state_dim   : dimension of the cross-hand recurrent state
    shuffle_opponents: whether to randomly permute opponent tokens
    """

    def __init__(
        self,
        d_model: int = 64,
        cards_heads: int = 4,
        cards_layers: int = 2,
        trunk_heads: int = 4,
        trunk_layers: int = 2,
        bets_hidden: int = 128,
        bets_in_dim: int = 512,  # 64 × 8
        obs_in_dim: int = 10,  # 10 self scalar features
        obs_hidden: int = 64,
        hand_state_dim: int = 64,
        game_state_dim: int = 64,
        shuffle_opponents: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.hand_state_dim = hand_state_dim
        self.game_state_dim = game_state_dim
        self.shuffle_opponents = shuffle_opponents

        # ── Encoders ─────────────────────────────────────────────────
        self.cards_encoder = CardsEncoder(
            d_card=d_model,
            n_heads=cards_heads,
            n_layers=cards_layers,
            out_dim=d_model,
        )

        self.bets_encoder = BetsNN(
            in_dim=bets_in_dim,
            hidden_dim=bets_hidden,
            out_dim=d_model,
        )

        self.obs_encoder = ObsNN(
            in_dim=obs_in_dim,
            hidden_dim=obs_hidden,
            out_dim=d_model,
        )

        # Recurrent state projections → d_model tokens
        self.hand_state_proj = nn.Linear(hand_state_dim, d_model)
        self.game_state_proj = nn.Linear(game_state_dim, d_model)

        # Opponent feature projection → d_model
        self.opp_proj = nn.Linear(OPP_FEAT_DIM, d_model)

        # ── Transformer trunk ─────────────────────────────────────────
        self.trunk = TransformerTrunk(
            d_model=d_model,
            n_heads=trunk_heads,
            n_layers=trunk_layers,
            n_opponents=MAX_OPPONENTS,
        )

        # ── Output heads ──────────────────────────────────────────────
        self.policy_head = nn.Linear(d_model, 3)  # fold / call / bet
        self.bet_mean_head = nn.Linear(d_model, 1)
        self.bet_std_head = nn.Linear(d_model, 1)

        # Recurrent state update heads
        self.hand_state_head = nn.Linear(d_model, hand_state_dim)
        self.game_state_head = nn.Linear(d_model, game_state_dim)

        # ── Internal state ────────────────────────────────────────────
        self._hand_state: torch.Tensor | None = None
        self._game_state: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def initialize_internal_state(self, batch_size: int = 1) -> None:
        """Call once before a new game session."""
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)
        self._game_state = torch.zeros(batch_size, self.game_state_dim, device=device)

    def new_hand(self, batch_size: int = 1) -> None:
        """Call between hands: reset hand_state, keep game_state."""
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.hand_state_dim, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        observation: Observation,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action_logits : [B, 3]      fold / call / bet
        bet_mean      : [B, 1]      ∈ (0, 1)
        bet_std       : [B, 1]      > 0
        """
        if self._hand_state is None or self._game_state is None:
            raise RuntimeError(
                "Internal state not initialised. "
                "Call initialize_internal_state() before the first forward()."
            )

        device = next(self.parameters()).device

        # ── Preprocess ───────────────────────────────────────────────
        p = preprocess_observation(observation)

        def _to(t):
            return t.to(device) if t is not None else None

        hole_cards = _to(p.hole_cards).unsqueeze(0)  # [1, 2, 2]
        board_cards = _to(p.board_cards).unsqueeze(0)  # [1, 5, 2]
        board_mask = _to(p.board_mask).unsqueeze(0)  # [1, 5]
        bets = _to(p.bets).unsqueeze(0)  # [1, 512]
        obs_scalars = _to(p.obs_scalars).unsqueeze(0)  # [1, 10]
        opp_vecs = _to(p.opp_vecs).unsqueeze(0)  # [1, 5, 8]
        opp_mask = _to(p.opp_mask).unsqueeze(0)  # [1, 5]

        # Use replayed internal state if this is a training replay
        hand_state = p.hand_state if p.hand_state is not None else self._hand_state
        game_state = p.game_state if p.game_state is not None else self._game_state

        # Store internal state into observation for trajectory logging
        observation.add_network_internal_state(
            {
                "hand": hand_state,
                "game": game_state,
            }
        )

        # ── Opponent shuffle ─────────────────────────────────────────
        if self.shuffle_opponents and self.training:
            perm = torch.randperm(MAX_OPPONENTS, device=device)
            opp_vecs = opp_vecs[:, perm, :]
            opp_mask = opp_mask[:, perm]

        # ── Encode each stream → d_model ─────────────────────────────
        f_cards = self.cards_encoder(
            hole_cards, board_cards, board_mask
        )  # [1, d_model]
        f_bets = self.bets_encoder(bets)  # [1, d_model]
        f_obs = self.obs_encoder(obs_scalars)  # [1, d_model]
        f_hand = self.hand_state_proj(hand_state)  # [1, d_model]
        f_game = self.game_state_proj(game_state)  # [1, d_model]
        f_opp = self.opp_proj(opp_vecs)  # [1, 5, d_model]

        # ── Transformer trunk → CLS ───────────────────────────────────
        cls = self.trunk(f_cards, f_bets, f_obs, f_hand, f_game, f_opp, opp_mask)
        # cls: [1, d_model]

        # ── Output heads ─────────────────────────────────────────────
        action_logits = self.policy_head(cls)  # [1, 3]
        bet_mean = torch.sigmoid(self.bet_mean_head(cls))  # [1, 1]
        bet_std = torch.exp(self.bet_std_head(cls)).clamp(min=1e-4, max=1.0)  # [1, 1]

        # ── Update internal state (detached — no gradient through time) ─
        new_hand_state = torch.tanh(self.hand_state_head(cls)).detach()
        new_game_state = torch.tanh(self.game_state_head(cls)).detach()

        if p.hand_state is None:  # not a replay → update live state
            self._hand_state = new_hand_state
            self._game_state = new_game_state

        return action_logits, bet_mean, bet_std
