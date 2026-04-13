import torch
import torch.nn as nn
from typing import Tuple

from nn.poker_net import PokerNet
from pokerenv.observation import Observation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBS_DIM = 57   # flat observation features (see _obs_to_tensor)
_LOG_DIM = 8    # features per hand-log step
_LOG_LEN = 32   # max timesteps in hand_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_tensor(obs: Observation) -> torch.Tensor:
    """
    Flattens the scalar / structural parts of an Observation into a 1-D
    float tensor of size _OBS_DIM (57).  The hand_log is handled separately
    by the GRU encoder.

    Breakdown:
        [0]       player_position
        [1]       player_stack
        [2]       player_money_in_pot
        [3]       bet_this_street
        [4]       street  (0-3)
        [5]       pot
        [6]       bet_to_match
        [7]       minimum_raise
        [8]       can_fold   (0/1)
        [9]       can_bet    (0/1)
        [10]      can_call   (0/1)
        [11]      bet_range_lower
        [12]      bet_range_upper
        [13-16]   hand card 0: suit, rank
        [17-20]   hand card 1: suit, rank
        [21-30]   table cards (up to 5) suit+rank pairs, 0-padded
        [31-56]   opponents (up to 5) x 6 features, 0-padded
    """
    parts: list[float] = [
        float(obs.player_position),
        float(obs.player_stack),
        float(obs.player_money_in_pot),
        float(obs.bet_this_street),
        float(obs.street),
        float(obs.pot),
        float(obs.bet_to_match),
        float(obs.minimum_raise),
        float(obs.actions.can_fold()),
        float(obs.actions.can_bet()),
        float(obs.actions.can_call()),
        float(obs.bet_range.lower_bound),
        float(obs.bet_range.upper_bound),
    ]

    # Hand cards (always 2)
    for card in obs.hand_cards.cards:
        parts += [float(card.suit), float(card.rank)]

    # Table cards (up to 5), zero-pad missing ones
    for card in obs.table_cards.cards:
        parts += [float(card.suit), float(card.rank)]
    for _ in range(5 - len(obs.table_cards.cards)):
        parts += [0.0, 0.0]

    # Other players (up to 5), zero-pad missing ones
    for other in obs.others:
        parts += [
            float(other.position),
            float(other.state),
            float(other.stack),
            float(other.money_in_pot),
            float(other.bet_this_street),
            float(other.is_all_in),
        ]
    for _ in range(5 - len(obs.others)):
        parts += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    assert len(parts) == _OBS_DIM, f"Expected {_OBS_DIM} features, got {len(parts)}"
    return torch.tensor(parts, dtype=torch.float32)


def _log_to_tensor(obs: Observation) -> torch.Tensor:
    """
    Converts hand_log (numpy [32, 4]) to a float tensor [32, 4].
    Unplayed slots are filled with -1.0 by the environment,
    which the GRU will learn to treat as padding.
    """
    return torch.from_numpy(obs.hand_log).float()   # [32, 4]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class FedeNet(PokerNet):
    """
    Stateless poker network with a GRU encoder for the hand action log.

    Architecture:
        ┌─ hand_log [32, 4]  ──► GRU ──► final hidden [gru_hidden]  ─┐
        │                                                              ├─► cat ──► backbone ──► heads
        └─ obs features [57] ─────────────────────────────────────────┘

    Backbone: 3 x (Linear -> ReLU)

    Heads:
        policy_head   -> 3  logits  (fold / call / bet)
        bet_mean_head -> 1  sigmoid -> [0, 1]
        bet_std_head  -> 1  softplus, clamped -> (1e-4, 1]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        gru_hidden: int = 64,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()

        # --- Hand-log encoder ---
        self.gru = nn.GRU(
            input_size=_LOG_DIM,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,       # input: [B, seq, features]
        )

        # --- Backbone ---
        backbone_in = _OBS_DIM + gru_hidden
        self.backbone = nn.Sequential(
            nn.Linear(backbone_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # --- Heads ---
        self.policy_head   = nn.Linear(hidden_dim, 3)
        self.bet_mean_head = nn.Linear(hidden_dim, 1)
        self.bet_std_head  = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------------------------
    # PokerNet interface
    # ------------------------------------------------------------------

    def preprocess(self, observation: Observation):
        """
        Returns a tuple (obs_vec, log_seq) ready for batching.
        """
        obs_vec = _obs_to_tensor(observation)      # [57]
        log_seq = _log_to_tensor(observation)      # [32, 4]
        return obs_vec, log_seq

    def initialize_internal_state(self) -> None:
        pass

    def new_hand(self) -> None:
        pass

    def forward(
        self, observation: Observation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            observation: a single Observation from the poker environment.

        Returns:
            action_logits : Tensor [3]  — raw logits for Categorical(logits=...)
            bet_mean      : Tensor [1]  — bet distribution mean, in [0, 1]
            bet_std       : Tensor [1]  — bet distribution std,  in (1e-4, 1]
        """
        device = next(self.parameters()).device

        # --- Encode observation features ---
        obs_vec = _obs_to_tensor(observation).to(device).unsqueeze(0)   # [1, 57]

        # --- Encode hand log with GRU ---
        log_seq = _log_to_tensor(observation).to(device).unsqueeze(0)   # [1, 32, 4]
        _, h_n = self.gru(log_seq)                                       # h_n: [layers, 1, gru_hidden]
        log_vec = h_n[-1]                                                # [1, gru_hidden] (last layer)

        # --- Fuse and run backbone ---
        x = torch.cat([obs_vec, log_vec], dim=1)    # [1, 57 + gru_hidden]
        h = self.backbone(x)                         # [1, hidden_dim]

        # --- Heads ---
        action_logits = self.policy_head(h)                   # [1, 3]
        bet_mean      = torch.sigmoid(self.bet_mean_head(h))  # [1, 1]
        bet_std       = torch.nn.functional.softplus(
            self.bet_std_head(h)
        ).clamp(1e-4, 1.0)                                   # [1, 1]

        return action_logits, bet_mean, bet_std
    
    def forward_batch(
        self,
        trajectory: list,  # list of (preprocessed_obs, Action)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched forward pass over a full trajectory for training replay.
        Must NOT update internal recurrent state.

        Returns
        -------
        action_logits : [N, 3]
        bet_mean      : [N, 1]
        bet_std       : [N, 1]
        """
        device = next(self.parameters()).device

        # Handle empty trajectory
        if len(trajectory) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return (
                zero * sum(p.sum() for p in self.parameters()),
                zero.unsqueeze(0).unsqueeze(0),
                zero.unsqueeze(0).unsqueeze(0),
            )

        obs_list = []
        log_list = []

        for (obs_data, _) in trajectory:
            obs_vec, log_seq = obs_data
            obs_list.append(obs_vec)
            log_list.append(log_seq)

        # Stack into batch
        obs_batch = torch.stack(obs_list).to(device)   # [N, 57]
        log_batch = torch.stack(log_list).to(device)   # [N, 32, 4]

        # --- GRU encoding ---
        _, h_n = self.gru(log_batch)                  # [layers, N, gru_hidden]
        log_vec = h_n[-1]                             # [N, gru_hidden]

        # --- Backbone ---
        x = torch.cat([obs_batch, log_vec], dim=1)    # [N, 57 + gru_hidden]
        h = self.backbone(x)                          # [N, hidden_dim]

        # --- Heads ---
        action_logits = self.policy_head(h)                   # [N, 3]
        bet_mean      = torch.sigmoid(self.bet_mean_head(h))  # [N, 1]
        bet_std       = torch.nn.functional.softplus(
            self.bet_std_head(h)
        ).clamp(1e-4, 1.0)                                   # [N, 1]

        return action_logits, bet_mean, bet_std
