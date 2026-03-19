"""
ui_player.py  (fixed)

Fix: _observation_to_dict now correctly detects check vs call.

In the Table._get_observation encoding:
  obs[1] = FOLD bit   (PlayerAction.FOLD  = 0  ->  action.value+1 = 1)
  obs[2] = BET  bit   (PlayerAction.BET   = 1  ->  action.value+1 = 2)
  obs[3] = CALL bit   (PlayerAction.CALL  = 2  ->  action.value+1 = 3)
  obs[4] = always 0   (no 4th action)

So in ActionsObservation(obs[1:5]):
  actions[0] = obs[1] = FOLD bit   <- NOT check!
  actions[1] = obs[2] = BET  bit
  actions[2] = obs[3] = CALL bit
  actions[3] = obs[4] = 0

can_check()  reads actions[0] which is really the FOLD bit — BUG in original.

Correct rule: a player can "check" when they CAN call AND bet_to_match == 0
(or they've already matched it with bet_this_street).  We compute this in
_observation_to_dict from the raw obs fields rather than trusting the helpers.
"""

import asyncio
import json
import threading
from typing import Optional

from pokerenv.observation import Observation
from pokerenv.common import PlayerAction
from pokerenv.action import Action
from pokerenv.player import Player

import torch


def _observation_to_dict(obs: Observation) -> dict:
    raw_can_fold = bool(obs.actions.actions[0])
    raw_can_bet = bool(obs.actions.actions[1])
    raw_can_call = bool(obs.actions.actions[2])

    can_check = raw_can_call and (
        float(obs.bet_to_match) == 0.0
        or float(obs.bet_this_street) >= float(obs.bet_to_match)
    )
    can_call = raw_can_call and not can_check

    # Use street to determine how many community cards have been dealt.
    # Filtering by (suit==0, rank==0) is wrong: the 2 of clubs has exactly
    # those values and would be dropped.
    n_table_cards = {0: 0, 1: 3, 2: 4, 3: 5}.get(int(obs.street), 0)

    return {
        "player_identifier": int(obs.player_identifier),
        "player_position": int(obs.player_position),
        "player_stack": float(obs.player_stack),
        "player_money_in_pot": float(obs.player_money_in_pot),
        "bet_this_street": float(obs.bet_this_street),
        "street": int(obs.street),
        "pot": float(obs.pot),
        "bet_to_match": float(obs.bet_to_match),
        "minimum_raise": float(obs.minimum_raise),
        "hand_cards": [
            {"suit": int(c.suit), "rank": int(c.rank)} for c in obs.hand_cards.cards
        ],
        "table_cards": [
            {"suit": int(c.suit), "rank": int(c.rank)}
            for c in obs.table_cards.cards[:n_table_cards]
        ],
        "actions": {
            "check": can_check,
            "call": can_call,
            "fold": raw_can_fold,
            "bet": raw_can_bet,
        },
        "bet_range": {
            "lower": float(obs.bet_range.lower_bound),
            "upper": float(obs.bet_range.upper_bound),
        },
        "others": [
            {
                "position": int(o.position),
                "state": int(o.state),
                "stack": float(o.stack),
                "money_in_pot": float(o.money_in_pot),
                "bet_this_street": float(o.bet_this_street),
                "is_all_in": bool(o.is_all_in),
            }
            for o in obs.others
        ],
    }


class UIPlayer(Player):
    is_ai = False

    def __init__(self, identifier: int, name: str, penalty: int = 0):
        super().__init__(identifier, name, penalty)
        self._websocket = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._action_event = threading.Event()
        self._pending_action: Optional[Action] = None
        self.disconnected: bool = False

    # ------------------------------------------------------------------
    # WebSocket plumbing
    # ------------------------------------------------------------------

    def attach_websocket(self, websocket, loop: asyncio.AbstractEventLoop):
        self._websocket = websocket
        self._loop = loop
        self.disconnected = False

    def detach_websocket(self):
        self._websocket = None
        self._loop = None
        self.disconnected = True
        self._action_event.set()

    @property
    def is_connected(self) -> bool:
        return self._websocket is not None

    # ------------------------------------------------------------------
    # Player interface
    # ------------------------------------------------------------------

    def get_action(self, observation: Observation) -> Action:
        if not self.is_connected:
            return self._make_fold(observation)

        self._pending_action = None
        self._action_event.clear()

        payload = json.dumps(
            {
                "type": "your_turn",
                "observation": _observation_to_dict(observation),
            }
        )
        try:
            asyncio.run_coroutine_threadsafe(
                self._websocket.send(payload), self._loop
            ).result(timeout=5)
        except Exception:
            self.disconnected = True
            return self._make_fold(observation)

        self._action_event.wait()

        if self.disconnected or self._pending_action is None:
            return self._make_fold(observation)

        return self._pending_action

    def receive_action_from_client(self, message: dict):
        """
        Expected: {"type": "action", "action_type": "check|call|fold|bet", "bet_amount": float}
        "check" -> CALL with bet_amount=0
        """
        action_type_str = message.get("action_type", "fold")
        bet_amount = float(message.get("bet_amount", 0.0))
        observation = message.get("_observation")

        _map = {
            "check": PlayerAction.CALL,
            "call": PlayerAction.CALL,
            "bet": PlayerAction.BET,
            "fold": PlayerAction.FOLD,
        }
        action_type = _map.get(action_type_str, PlayerAction.FOLD)

        if action_type in (PlayerAction.FOLD, PlayerAction.CALL):
            bet_amount = 0.0

        d = torch.tensor(action_type.value, dtype=torch.long)
        bet_tensor = torch.tensor(0.0)

        if observation is not None and action_type is PlayerAction.BET:
            rng = observation.bet_range.upper_bound - observation.bet_range.lower_bound
            if rng > 0:
                bet_tensor = torch.tensor(
                    (bet_amount - observation.bet_range.lower_bound) / rng
                ).clamp(0.0, 1.0)

        self._pending_action = Action(
            action_type=action_type,
            action_tensor=d,
            observation=observation,
            bet_amount=bet_amount,
            bet_tensor=bet_tensor,
        )
        self._action_event.set()

    def new_hand(self):
        pass

    def reset(self):
        super().reset()
        self._pending_action = None
        self._action_event.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_fold(self, observation: Observation) -> Action:
        d = torch.tensor(PlayerAction.FOLD.value, dtype=torch.long)
        return Action(
            action_type=PlayerAction.FOLD,
            action_tensor=d,
            observation=observation,
            bet_amount=0.0,
            bet_tensor=torch.tensor(0.0),
        )

    async def send_message(self, payload: dict):
        if self._websocket is not None:
            try:
                await self._websocket.send(json.dumps(payload))
            except Exception:
                pass
