from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from nn.poker_net import PokerNet
from pokerenv.observation import Observation


def _clamp_index(value: float, upper: int) -> int:
    return max(0, min(int(value), upper))


def _log1p_features(values: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(values, min=0.0))


class _CardEncoder(nn.Module):
    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.rank_embedding = nn.Embedding(14, 16)
        self.suit_embedding = nn.Embedding(5, 8)
        self.slot_embedding = nn.Embedding(7, 8)
        self.token_projection = nn.Linear(33, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("slot_ids", torch.arange(7), persistent=False)

    def forward(
        self,
        rank_ids: torch.Tensor,
        suit_ids: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rank_ids.shape[0]
        slot_ids = self.slot_ids.unsqueeze(0).expand(batch_size, -1)

        tokens = torch.cat(
            [
                self.rank_embedding(rank_ids),
                self.suit_embedding(suit_ids),
                self.slot_embedding(slot_ids),
                presence,
            ],
            dim=-1,
        )
        tokens = self.token_projection(tokens)

        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.output_norm(encoded[:, 0, :])


class _TableStateEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.street_embedding = nn.Embedding(4, 8)
        self.position_embedding = nn.Embedding(6, 8)
        self.input_norm = nn.LayerNorm(27)
        self.net = nn.Sequential(
            nn.Linear(27, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

    def forward(
        self,
        legal_actions: torch.Tensor,
        continuous_features: torch.Tensor,
        street_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                legal_actions,
                _log1p_features(continuous_features),
                self.street_embedding(street_ids),
                self.position_embedding(position_ids),
            ],
            dim=1,
        )
        return self.net(self.input_norm(features))


class _OpponentEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(7, 8)
        self.state_embedding = nn.Embedding(4, 4)
        self.row_mlp = nn.Sequential(
            nn.Linear(17, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
        )
        self.query_projection = nn.Linear(64, 64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)

    def forward(
        self,
        opponent_rows: torch.Tensor,
        card_ctx: torch.Tensor,
        table_ctx: torch.Tensor,
    ) -> torch.Tensor:
        present_mask = opponent_rows.abs().sum(dim=-1) > 0

        position_ids = torch.zeros(
            opponent_rows.shape[:2], dtype=torch.long, device=opponent_rows.device
        )
        state_ids = torch.zeros_like(position_ids)

        if present_mask.any():
            position_ids[present_mask] = (
                opponent_rows[..., 0][present_mask].long().clamp(0, 5) + 1
            )
            state_ids[present_mask] = (
                opponent_rows[..., 1][present_mask].long().clamp(0, 2) + 1
            )

        continuous = opponent_rows[..., 2:6].clone()
        continuous[..., :3] = _log1p_features(continuous[..., :3])

        row_features = torch.cat(
            [
                self.position_embedding(position_ids),
                self.state_embedding(state_ids),
                continuous,
                present_mask.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        row_features = self.row_mlp(row_features)

        query = self.query_projection(card_ctx + table_ctx).unsqueeze(1)
        pooled = []
        for batch_index in range(row_features.shape[0]):
            if not present_mask[batch_index].any():
                pooled.append(row_features.new_zeros(1, row_features.shape[-1]))
                continue

            attn_output, _ = self.attention(
                query[batch_index : batch_index + 1],
                row_features[batch_index : batch_index + 1],
                row_features[batch_index : batch_index + 1],
                key_padding_mask=~present_mask[batch_index : batch_index + 1],
            )
            pooled.append(attn_output[:, 0, :])

        return torch.cat(pooled, dim=0)


class _HandHistoryEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_embedding = nn.Embedding(7, 8)
        self.action_embedding = nn.Embedding(4, 8)
        self.street_embedding = nn.Embedding(5, 4)
        self.row_projection = nn.Linear(21, 32)
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)

    def forward(
        self,
        actor_ids: torch.Tensor,
        action_ids: torch.Tensor,
        street_ids: torch.Tensor,
        bet_fraction: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        row_features = torch.cat(
            [
                self.actor_embedding(actor_ids),
                self.action_embedding(action_ids),
                self.street_embedding(street_ids),
                bet_fraction,
            ],
            dim=-1,
        )
        row_features = self.row_projection(row_features)

        lengths = valid_mask.sum(dim=1).long()
        if lengths.max().item() == 0:
            return row_features.new_zeros(row_features.shape[0], 64)

        packed = pack_padded_sequence(
            row_features,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        history_ctx = hidden[-1]
        history_ctx[lengths == 0] = 0.0
        return history_ctx


class DaniNet(PokerNet):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 64

        self.card_encoder = _CardEncoder()
        self.table_encoder = _TableStateEncoder()
        self.opponent_encoder = _OpponentEncoder()
        self.history_encoder = _HandHistoryEncoder()

        self.fusion_norm = nn.LayerNorm(384)
        self.fusion = nn.Sequential(
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
        )

        self.hand_gru = nn.GRUCell(128, self.state_dim)
        self.game_gru = nn.GRUCell(128, self.state_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.bet_mean_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.bet_std_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self._hand_state: torch.Tensor | None = None
        self._game_state: torch.Tensor | None = None

    def initialize_internal_state(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.state_dim, device=device)
        self._game_state = torch.zeros(batch_size, self.state_dim, device=device)

    def new_hand(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        self._hand_state = torch.zeros(batch_size, self.state_dim, device=device)
        if self._game_state is None:
            self._game_state = torch.zeros(batch_size, self.state_dim, device=device)

    def forward(
        self, observation: Observation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        batch_size = 1

        if observation.is_replay:
            hand_state = self._prepare_state_tensor(
                observation.network_internal_state["hand"], device
            )
            game_state = self._prepare_state_tensor(
                observation.network_internal_state["game"], device
            )
            replay_mode = True
        else:
            if self._hand_state is None or self._game_state is None:
                raise RuntimeError("Internal state not initialized.")

            hand_state = self._hand_state
            game_state = self._game_state
            replay_mode = False

            observation.add_network_internal_state(
                {
                    "hand": hand_state.detach().cpu(),
                    "game": game_state.detach().cpu(),
                }
            )

        rank_ids, suit_ids, presence = self._extract_card_inputs(observation, device)
        legal_actions, continuous_features, street_ids, position_ids = (
            self._extract_table_inputs(observation, device)
        )
        opponent_rows = self._extract_opponent_inputs(observation, device)
        actor_ids, action_ids, history_street_ids, bet_fraction, valid_mask = (
            self._extract_history_inputs(observation, device)
        )

        card_ctx = self.card_encoder(rank_ids, suit_ids, presence)
        table_ctx = self.table_encoder(
            legal_actions, continuous_features, street_ids, position_ids
        )
        opponent_ctx = self.opponent_encoder(opponent_rows, card_ctx, table_ctx)
        history_ctx = self.history_encoder(
            actor_ids, action_ids, history_street_ids, bet_fraction, valid_mask
        )

        fused = torch.cat(
            [card_ctx, table_ctx, opponent_ctx, history_ctx, hand_state, game_state],
            dim=1,
        )
        fused = self.fusion(self.fusion_norm(fused))

        new_hand_state = self.hand_gru(fused, hand_state)
        new_game_state = self.game_gru(fused, game_state)

        if not replay_mode:
            self._hand_state = new_hand_state.detach()
            self._game_state = new_game_state.detach()

        action_logits = self._mask_action_logits(
            self.policy_head(fused), legal_actions.bool()
        )
        bet_mean = torch.sigmoid(self.bet_mean_head(fused))
        bet_std = F.softplus(self.bet_std_head(fused)).clamp(min=1e-4, max=1.0)

        if batch_size == 1:
            return action_logits.squeeze(0), bet_mean.squeeze(0), bet_std.squeeze(0)
        return action_logits, bet_mean, bet_std

    def _prepare_state_tensor(
        self, state: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        state = state.to(device=device, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return state

    def _extract_card_inputs(
        self, observation: Observation, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rank_ids = torch.zeros((1, 7), dtype=torch.long, device=device)
        suit_ids = torch.zeros((1, 7), dtype=torch.long, device=device)
        presence = torch.zeros((1, 7, 1), dtype=torch.float32, device=device)

        street = _clamp_index(float(observation.street), 3)
        board_cards_visible = [0, 3, 4, 5][street]

        all_cards = list(observation.hand_cards.cards) + list(observation.table_cards.cards)
        for slot_index, card in enumerate(all_cards):
            is_present = slot_index < 2 or (slot_index - 2) < board_cards_visible
            if not is_present:
                continue

            rank_ids[0, slot_index] = _clamp_index(float(card.rank), 12) + 1
            suit_ids[0, slot_index] = _clamp_index(float(card.suit), 3) + 1
            presence[0, slot_index, 0] = 1.0

        return rank_ids, suit_ids, presence

    def _extract_table_inputs(
        self, observation: Observation, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        legal_actions = torch.tensor(
            [
                [
                    float(observation.actions.can_fold()),
                    float(observation.actions.can_bet()),
                    float(observation.actions.can_call()),
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        continuous = torch.tensor(
            [
                [
                    float(observation.bet_range.lower_bound),
                    float(observation.bet_range.upper_bound),
                    float(observation.player_stack),
                    float(observation.player_money_in_pot),
                    float(observation.bet_this_street),
                    float(observation.pot),
                    float(observation.bet_to_match),
                    float(observation.minimum_raise),
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        street_ids = torch.tensor(
            [_clamp_index(float(observation.street), 3)],
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.tensor(
            [_clamp_index(float(observation.player_position), 5)],
            dtype=torch.long,
            device=device,
        )
        return legal_actions, continuous, street_ids, position_ids

    def _extract_opponent_inputs(
        self, observation: Observation, device: torch.device
    ) -> torch.Tensor:
        rows = torch.zeros((1, 5, 6), dtype=torch.float32, device=device)
        for index, other in enumerate(observation.others[:5]):
            rows[0, index] = torch.tensor(
                [
                    float(other.position),
                    float(other.state),
                    float(other.stack),
                    float(other.money_in_pot),
                    float(other.bet_this_street),
                    float(other.is_all_in),
                ],
                dtype=torch.float32,
                device=device,
            )
        return rows

    def _extract_history_inputs(
        self, observation: Observation, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_log = torch.as_tensor(observation.hand_log, dtype=torch.float32, device=device)
        if hand_log.dim() == 2:
            hand_log = hand_log.unsqueeze(0)

        valid_mask = hand_log[..., 0] != -1.0

        actor_ids = torch.zeros(hand_log.shape[:2], dtype=torch.long, device=device)
        action_ids = torch.zeros_like(actor_ids)
        street_ids = torch.zeros_like(actor_ids)

        actor_ids[valid_mask] = hand_log[..., 0][valid_mask].long().clamp(0, 5) + 1
        action_ids[valid_mask] = hand_log[..., 1][valid_mask].long().clamp(0, 2) + 1
        street_ids[valid_mask] = hand_log[..., 3][valid_mask].long().clamp(0, 3) + 1

        bet_fraction = hand_log[..., 2].unsqueeze(-1)
        bet_fraction[~valid_mask.unsqueeze(-1)] = 0.0

        return actor_ids, action_ids, street_ids, bet_fraction, valid_mask

    def _mask_action_logits(
        self, action_logits: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        if legal_mask.any(dim=1).all():
            return action_logits.masked_fill(~legal_mask, -1e9)

        masked = action_logits.clone()
        valid_rows = legal_mask.any(dim=1)
        masked[valid_rows] = masked[valid_rows].masked_fill(~legal_mask[valid_rows], -1e9)
        return masked
