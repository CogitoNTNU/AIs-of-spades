import numpy as np
import torch
import os
import sys

try:
    from nn.miheer_net import MiheerNet
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from nn.miheer_net import MiheerNet


# Minimal stand-in classes to mimic pokerenv.observation structures
class _Card:
    def __init__(self, suit: int, rank: int) -> None:
        self.suit = suit
        self.rank = rank


class _CardCollection:
    def __init__(self, cards):
        self.cards = cards


class FakeObservation:
    """
    Minimal observation compatible with MiheerNet preprocess expectations.
    """

    def __init__(self, hand_log: np.ndarray):
        # two hand cards
        self.hand_cards = _CardCollection(
            [
                _Card(suit=0, rank=12),  # Ace of suit 0
                _Card(suit=1, rank=11),  # King of suit 1
            ]
        )
        # up to 5 community cards
        self.table_cards = _CardCollection(
            [
                _Card(suit=2, rank=10),
                _Card(suit=3, rank=9),
                _Card(suit=0, rank=8),
                _Card(suit=1, rank=7),
                _Card(suit=2, rank=6),
            ]
        )

        self.hand_log = hand_log
        self.is_replay = False
        self.network_internal_state = {}

    def add_network_internal_state(self, state_dict):
        # For replay compatibility; store the provided state.
        self.network_internal_state.update(state_dict)


def make_fake_observation(seq_len: int = 64, feat_dim: int = 8) -> FakeObservation:
    # Simple synthetic betting log: ramping values
    hand_log = np.linspace(0.0, 1.0, num=seq_len * feat_dim, dtype=np.float32).reshape(
        seq_len, feat_dim
    )
    return FakeObservation(hand_log)


def run_smoke(device: torch.device | str = "cpu") -> None:
    device = torch.device(device)
    torch.manual_seed(0)

    net = MiheerNet().to(device)
    net.initialize_internal_state(batch_size=1)

    obs = make_fake_observation()
    action_logits, bet_mean, bet_std = net(obs)

    assert action_logits.shape == (1, 3), f"Unexpected action_logits shape {action_logits.shape}"
    assert bet_mean.shape == (1, 1), f"Unexpected bet_mean shape {bet_mean.shape}"
    assert bet_std.shape == (1, 1), f"Unexpected bet_std shape {bet_std.shape}"

    print("Smoke test passed.")
    print("action_logits:", action_logits.detach().cpu().numpy())
    print("bet_mean:", bet_mean.detach().cpu().numpy())
    print("bet_std:", bet_std.detach().cpu().numpy())
    print("internal hand_state shape:", net._hand_state.shape)
    print("internal game_state shape:", net._game_state.shape)


if __name__ == "__main__":
    run_smoke("cuda" if torch.cuda.is_available() else "cpu")
