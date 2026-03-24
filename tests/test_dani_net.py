from pathlib import Path

import numpy as np
import torch

from nn.dani_net import DaniNet
from nn.model_classes import MODEL_CLASSES
from pokerenv.observation import Observation
from training.player_agent import PlayerAgent
from ui.ui_agent_player import AIPlayer


SUITS = [1.0, 2.0, 4.0, 8.0]


def _make_observation(
    *,
    street: int = 0,
    legal_actions: tuple[int, int, int] = (1, 1, 1),
) -> Observation:
    obs = np.zeros(58, dtype=np.float32)
    obs[0] = 0
    obs[1:4] = np.asarray(legal_actions, dtype=np.float32)
    obs[4:6] = np.asarray([0.1, 0.8], dtype=np.float32)
    obs[6] = 2

    obs[7:11] = np.asarray([SUITS[0], 12, SUITS[1], 11], dtype=np.float32)
    obs[11] = 100
    obs[12] = 5
    obs[13] = 2
    obs[14] = street

    board_cards = [
        (SUITS[2], 8),
        (SUITS[3], 7),
        (SUITS[0], 6),
        (SUITS[1], 4),
        (SUITS[2], 2),
    ]
    for index, (suit, rank) in enumerate(board_cards):
        base = 15 + index * 2
        obs[base : base + 2] = np.asarray([suit, rank], dtype=np.float32)

    obs[25] = 30
    obs[26] = 2
    obs[27] = 4

    others = [
        [0, 1, 95, 5, 1, 0],
        [1, 1, 120, 10, 2, 0],
    ]
    for index, row in enumerate(others):
        base = 28 + index * 6
        obs[base : base + 6] = np.asarray(row, dtype=np.float32)

    hand_log = np.full((32, 4), -1.0, dtype=np.float32)
    hand_log[0] = np.asarray([0, 1, 0.5, 0], dtype=np.float32)
    hand_log[1] = np.asarray([1, 2, 0.25, 0], dtype=np.float32)
    if street >= 1:
        hand_log[2] = np.asarray([0, 1, 0.3, 1], dtype=np.float32)

    return Observation(obs, hand_log)


def test_dani_net_forward_returns_finite_outputs():
    model = DaniNet()
    model.initialize_internal_state()

    action_logits, bet_mean, bet_std = model.forward(_make_observation())

    assert action_logits.shape == (3,)
    assert bet_mean.numel() == 1
    assert bet_std.numel() == 1
    assert torch.isfinite(action_logits).all()
    assert torch.isfinite(bet_mean).all()
    assert torch.isfinite(bet_std).all()


def test_dani_net_handles_empty_and_river_observations():
    model = DaniNet()
    model.initialize_internal_state()

    empty_logits, empty_mean, empty_std = model.forward(Observation.empty())
    river_logits, river_mean, river_std = model.forward(_make_observation(street=3))

    assert torch.isfinite(empty_logits).all()
    assert torch.isfinite(empty_mean).all()
    assert torch.isfinite(empty_std).all()
    assert torch.isfinite(river_logits).all()
    assert torch.isfinite(river_mean).all()
    assert torch.isfinite(river_std).all()


def test_dani_net_masks_invalid_actions():
    model = DaniNet()
    model.initialize_internal_state()

    action_logits, _, _ = model.forward(_make_observation(legal_actions=(0, 0, 1)))

    assert action_logits[0].item() <= -1e8
    assert action_logits[1].item() <= -1e8
    assert torch.isfinite(action_logits[2])


def test_dani_net_state_reset_semantics():
    model = DaniNet()
    model.initialize_internal_state()

    assert torch.allclose(model._hand_state, torch.zeros_like(model._hand_state))
    assert torch.allclose(model._game_state, torch.zeros_like(model._game_state))

    model._hand_state = torch.ones_like(model._hand_state)
    model._game_state = 2 * torch.ones_like(model._game_state)

    model.new_hand()

    assert torch.allclose(model._hand_state, torch.zeros_like(model._hand_state))
    assert torch.allclose(model._game_state, 2 * torch.ones_like(model._game_state))

    model.initialize_internal_state()

    assert torch.allclose(model._hand_state, torch.zeros_like(model._hand_state))
    assert torch.allclose(model._game_state, torch.zeros_like(model._game_state))


def test_dani_net_replay_does_not_mutate_module_memory():
    model = DaniNet()
    model.initialize_internal_state()
    model._hand_state = torch.full_like(model._hand_state, 7.0)
    model._game_state = torch.full_like(model._game_state, 9.0)

    obs = _make_observation(street=2)
    obs.add_network_internal_state(
        {
            "hand": torch.full((1, 64), 3.0),
            "game": torch.full((1, 64), 4.0),
        }
    )

    model.forward(obs)

    assert torch.allclose(model._hand_state, torch.full((1, 64), 7.0))
    assert torch.allclose(model._game_state, torch.full((1, 64), 9.0))


def test_dani_net_state_dict_round_trip():
    model = DaniNet()
    model.initialize_internal_state()
    cloned = DaniNet()
    cloned.load_state_dict(model.state_dict())

    original = model.forward(_make_observation(street=1))
    cloned.initialize_internal_state()
    copied = cloned.forward(_make_observation(street=1))

    for original_tensor, copied_tensor in zip(original, copied):
        assert copied_tensor.shape == original_tensor.shape


def test_model_classes_registers_dani_net():
    assert MODEL_CLASSES["DaniNet"] is DaniNet


def test_player_agent_smoke_with_dani_net():
    model = DaniNet()
    model.initialize_internal_state()
    agent = PlayerAgent(0, "UGO", 0, model)

    action = agent.get_action(_make_observation(street=1))

    assert action.action_tensor.dim() == 0
    assert action.bet_tensor.numel() == 1


def test_ai_player_smoke_with_dani_net(tmp_path: Path):
    model = DaniNet()
    model.initialize_internal_state()
    weights_path = tmp_path / "dani_net.pt"
    torch.save(model.state_dict(), weights_path)

    ai_player = AIPlayer(0, "UGO", DaniNet, str(weights_path))
    action = ai_player.get_action(_make_observation(street=3))

    assert action.action_tensor.dim() == 0
    assert action.bet_tensor.numel() == 1
