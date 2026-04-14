import os
import sys

import numpy as np
import torch

try:
    from nn.miheer_net2 import MiheerNet2
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from nn.miheer_net2 import MiheerNet2


class _Card:
    def __init__(self, suit: int, rank: int) -> None:
        self.suit = suit
        self.rank = rank


class _CardCollection:
    def __init__(self, cards):
        self.cards = cards


class _Other:
    def __init__(self, position, state, stack, money_in_pot, bet_this_street, is_all_in):
        self.position = position
        self.state = state
        self.stack = stack
        self.money_in_pot = money_in_pot
        self.bet_this_street = bet_this_street
        self.is_all_in = is_all_in


class _Actions:
    def can_fold(self):
        return True

    def can_bet(self):
        return True

    def can_call(self):
        return True


class _BetRange:
    def __init__(self, lower_bound: float = 0.0, upper_bound: float = 100.0):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class FakeObservation:
    def __init__(self, hand_log: np.ndarray):
        self.player_identifier = 0
        self.player_position = 0
        self.player_stack = 100.0
        self.player_money_in_pot = 0.0
        self.bet_this_street = 0.0
        self.street = 0
        self.pot = 1.5
        self.bet_to_match = 0.0
        self.minimum_raise = 2.0

        self.actions = _Actions()
        self.bet_range = _BetRange()

        self.hand_cards = _CardCollection([
            _Card(0, 12),
            _Card(1, 11),
        ])
        self.table_cards = _CardCollection([
            _Card(2, 10),
            _Card(3, 9),
            _Card(0, 8),
            _Card(1, 7),
            _Card(2, 6),
        ])

        self.others = [
            _Other(1, 1, 100.0, 0.0, 0.0, 0.0),
            _Other(2, 1, 95.0, 1.0, 1.0, 0.0),
        ]

        self.hand_log = hand_log
        self.is_replay = False
        self.network_internal_state = {}

    def add_network_internal_state(self, state_dict):
        self.network_internal_state.update(state_dict)


def make_fake_observation(seq_len: int = 64, feat_dim: int = 8) -> FakeObservation:
    hand_log = np.full((seq_len, feat_dim), -1.0, dtype=np.float32)
    hand_log[:8] = np.linspace(0.0, 1.0, num=8 * feat_dim, dtype=np.float32).reshape(8, feat_dim)
    return FakeObservation(hand_log)


def run_smoke(device: torch.device | str = "cpu") -> None:
    device = torch.device(device)
    torch.manual_seed(0)

    net = MiheerNet2().to(device)
    net.initialize_internal_state(batch_size=1)

    obs = make_fake_observation()

    action_logits, bet_mean, bet_std = net(obs)
    assert action_logits.shape == (1, 3), f"Unexpected action_logits shape {action_logits.shape}"
    assert bet_mean.shape == (1, 1), f"Unexpected bet_mean shape {bet_mean.shape}"
    assert bet_std.shape == (1, 1), f"Unexpected bet_std shape {bet_std.shape}"

    pre = net.preprocess(obs)
    action_logits_b, bet_mean_b, bet_std_b = net.forward_batch([(pre, None)])
    assert action_logits_b.shape == (1, 3), f"Unexpected batch action_logits shape {action_logits_b.shape}"
    assert bet_mean_b.shape == (1, 1), f"Unexpected batch bet_mean shape {bet_mean_b.shape}"
    assert bet_std_b.shape == (1, 1), f"Unexpected batch bet_std shape {bet_std_b.shape}"

    assert net._hand_state is not None
    assert net._game_state is not None
    assert net._opp_memory is not None

    print("MiheerNet2 smoke test passed.")
    print("forward action_logits:", action_logits.detach().cpu().numpy())
    print("forward bet_mean:", bet_mean.detach().cpu().numpy())
    print("forward bet_std:", bet_std.detach().cpu().numpy())
    print("hand_state shape:", tuple(net._hand_state.shape))
    print("game_state shape:", tuple(net._game_state.shape))
    print("opp_memory shape:", tuple(net._opp_memory.shape))


if __name__ == "__main__":
    run_smoke("cuda" if torch.cuda.is_available() else "cpu")
