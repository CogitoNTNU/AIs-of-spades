import asyncio
import json
import logging
import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional

import websockets
import torch
import math
from treys import Card as TreysCard
import numpy as np


from pokerenv.table import Table
from pokerenv.observation import Observation
from pokerenv.common import PlayerState, PlayerAction
from pokerenv.action import Action

from .ui_player import UIPlayer, _observation_to_dict
from .html_client import _HTML_TEMPLATE

log = logging.getLogger("UITableManager")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _HTTPHandler(BaseHTTPRequestHandler):
    html: str = ""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_HTTPHandler.html.encode())

    def log_message(self, *args):
        pass


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------


class UITableManager:
    def __init__(
        self,
        n_seats: int = 4,
        host: str = "0.0.0.0",
        ws_port: int = 8765,
        http_port: int = 8080,
        total_hands: int = 20,
        stack_low: int = 50,
        stack_high: int = 200,
    ):
        self.n_seats = n_seats
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port
        self.total_hands = total_hands
        self.stack_low = stack_low
        self.stack_high = stack_high

        self.players: Dict[int, UIPlayer] = {}
        self._name_to_seat: Dict[str, int] = {}
        self._last_obs: Dict[int, Observation] = {}
        self._game_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._game_running: bool = False
        self._hand_ack_events: Dict[int, threading.Event] = {}

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    # ------------------------------------------------------------------
    # Async server
    # ------------------------------------------------------------------

    async def _serve(self):
        local_ip = _get_local_ip()
        log.info("WebSocket  ->  ws://%s:%d", self.host, self.ws_port)
        log.info("HTTP       ->  http://%s:%d", self.host, self.http_port)
        log.info(">>> Connect at: http://%s:%d <<<", local_ip, self.http_port)
        self._start_http_server()

        async with websockets.serve(self._on_connect, self.host, self.ws_port):
            await self._lobby_loop()

    async def _lobby_loop(self):
        while True:
            log.info("Waiting for %d player(s)...", self.n_seats)
            while len(self.players) < self.n_seats:
                await asyncio.sleep(0.3)

            log.info("All seats filled — starting game")
            self._game_running = True
            await asyncio.get_event_loop().run_in_executor(None, self._run_game)
            self._game_running = False

            self.players.clear()
            self._name_to_seat.clear()
            self._last_obs.clear()

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _on_connect(self, websocket):
        seat = None
        try:
            async for raw in websocket:
                msg = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "join":
                    name = msg.get("name", "Player").strip()
                    seat, reconnected = self._assign_or_reconnect(websocket, name)

                    if seat is None:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": "Table is full or name already taken.",
                                }
                            )
                        )
                        return

                    player = self.players[seat]
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "welcome",
                                "seat": seat,
                                "name": player.name,
                                "total_hands": self.total_hands,
                            }
                        )
                    )

                    if reconnected:
                        log.info("Seat %d RECONNECTED as %s", seat, player.name)
                        await self._send_reconnect_state(websocket, seat)
                    else:
                        log.info("Seat %d  ->  %s", seat, player.name)

                    await self._broadcast_lobby_status()

                elif mtype == "action" and seat is not None:
                    obs = self._last_obs.get(seat)
                    msg["_observation"] = obs
                    self.players[seat].receive_action_from_client(msg)

                elif mtype == "hand_ack" and seat is not None:
                    event = self._hand_ack_events.get(seat)
                    if event:
                        event.set()
        except websockets.exceptions.ConnectionClosed:
            log.info("Seat %s disconnected", seat)
        finally:
            if seat is not None and seat in self.players:
                self.players[seat].detach_websocket()

    def _assign_or_reconnect(self, websocket, name: str):
        with self._game_lock:
            name_key = name.lower()

            if self._game_running and name_key in self._name_to_seat:
                seat = self._name_to_seat[name_key]
                player = self.players.get(seat)
                if player is not None and not player.is_connected:
                    player.attach_websocket(websocket, self._loop)
                    return seat, True
                else:
                    return None, False

            if self._game_running:
                return None, False

            taken = set(self.players.keys())
            for seat in range(self.n_seats):
                if seat not in taken:
                    player = UIPlayer(seat, name)
                    player.attach_websocket(websocket, self._loop)
                    self.players[seat] = player
                    self._name_to_seat[name.lower()] = seat
                    return seat, False

        return None, False

    async def _send_reconnect_state(self, websocket, seat: int):
        obs = self._last_obs.get(seat)
        if obs is not None:
            agents = [self.players[i] for i in sorted(self.players)]
            payload = _build_table_update(obs, agents)
            try:
                await websocket.send(json.dumps(payload))
            except Exception:
                pass

    async def _broadcast_lobby_status(self):
        seated = [self.players[s].name for s in sorted(self.players)]
        needed = self.n_seats - len(seated)
        payload = json.dumps({"type": "waiting", "seated": seated, "needed": needed})
        for player in self.players.values():
            if player.is_connected:
                try:
                    await player._websocket.send(payload)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Game loop  (thread-pool executor)
    # ------------------------------------------------------------------

    def _run_game(self):
        agents = [self.players[i] for i in sorted(self.players)]
        n = len(agents)

        table = Table(
            n,
            players=agents,
            stack_low=self.stack_low,
            stack_high=self.stack_high,
            hand_history_location="hands/",
            hand_history_enabled=True,
            invalid_action_penalty=0,
        )
        table.seed(None)
        table.reset()

        # Track how many hh lines we've already pushed so we only push new ones
        _hh_pushed: int = 0

        def _push_new_hh_lines():
            nonlocal _hh_pushed
            history = table.hh.history
            new_lines = history[_hh_pushed:]
            if new_lines:
                self._broadcast_sync({"type": "hand_log", "lines": new_lines})
                _hh_pushed = len(history)

        for hand_index in range(self.total_hands):
            log.info("Hand %d / %d", hand_index + 1, self.total_hands)

            _hh_pushed = 0  # Reset for each hand

            for agent in agents:
                agent.new_hand()

            obs_array = table.reset_hand()
            obs = Observation(obs_array, table.hand_log)

            self._broadcast_sync(_build_table_update(obs, agents))
            self._broadcast_sync(_build_turn_indicator(obs, agents))
            self._broadcast_player_states(agents)
            _push_new_hh_lines()

            rewards = np.zeros(n)

            while True:
                acting_seat = int(obs.player_identifier)

                if acting_seat >= n:
                    log.warning(
                        "player_identifier %d out of range — ending hand", acting_seat
                    )
                    break

                acting_agent = agents[acting_seat]

                if acting_agent.state is not PlayerState.ACTIVE or acting_agent.all_in:
                    log.warning(
                        "Table asked inactive player %d — forcing _end_hand",
                        acting_seat,
                    )
                    table.hand_is_over = True
                    table._end_hand()
                    rewards = np.asarray([p.get_reward() for p in sorted(agents)])
                    break

                self._last_obs[acting_seat] = obs
                self._push_your_turn(acting_seat, obs)
                action = acting_agent.get_action(obs)
                obs_array, rewards, done = table.step(action)

                # Push any new HH lines produced by this action
                _push_new_hh_lines()

                if done:
                    break

                obs = Observation(obs_array, table.hand_log)
                self._broadcast_sync(_build_table_update(obs, agents))
                self._broadcast_sync(_build_turn_indicator(obs, agents))

            # Broadcast board finale con community cards complete
            community = []
            for c in table.street_mgr.cards:
                suit_bitmask = TreysCard.get_suit_int(c)
                rank_idx = TreysCard.get_rank_int(c)
                suit_idx = int(math.log2(suit_bitmask)) if suit_bitmask > 0 else 0
                community.append({"suit": suit_idx, "rank": rank_idx})

            self._broadcast_sync(
                {
                    "type": "table_update",
                    "street": int(table.street_mgr.street),
                    "pot": float(table.pot_mgr.pot),
                    "bet_to_match": 0.0,
                    "table_cards": community,
                    "all_players": [
                        {
                            "seat": i,
                            "name": agents[i].name,
                            "position": int(agents[i].position),
                            "state": agents[i].state.value,
                            "stack": float(agents[i].stack),
                            "money_in_pot": float(agents[i].money_in_pot),
                            "bet_this_street": float(agents[i].bet_this_street),
                            "is_all_in": bool(agents[i].all_in),
                        }
                        for i in range(n)
                    ],
                    # Keep legacy "others" for compatibility
                    "others": [
                        {
                            "position": int(agents[i].position),
                            "state": agents[i].state.value,
                            "stack": float(agents[i].stack),
                            "money_in_pot": float(agents[i].money_in_pot),
                            "bet_this_street": float(agents[i].bet_this_street),
                            "is_all_in": bool(agents[i].all_in),
                        }
                        for i in range(n)
                    ],
                }
            )

            # Push final HH lines (showdown, summary)
            _push_new_hh_lines()

            # Converti le carte dello showdown
            showdown = {}
            for player_name, cards in table.showdown_cards.items():
                hand = []
                for c in cards:
                    suit_bitmask = TreysCard.get_suit_int(c)
                    rank_idx = TreysCard.get_rank_int(c)
                    suit_idx = int(math.log2(suit_bitmask)) if suit_bitmask > 0 else 0
                    hand.append({"suit": suit_idx, "rank": rank_idx})
                showdown[player_name] = hand

            # Stato reale dei giocatori dagli agenti
            final_players = [
                {
                    "name": agents[i].name,
                    "seat": i,
                    "stack": float(agents[i].stack),
                    "state": agents[i].state.value,
                    "money_in_pot": float(agents[i].money_in_pot),
                    "bet_this_street": float(agents[i].bet_this_street),
                    "is_all_in": bool(agents[i].all_in),
                }
                for i in range(n)
            ]

            reward_dict = {
                agents[i].name: float(rewards[i])
                for i in range(n)
                if rewards[i] is not None
            }

            self._broadcast_sync(
                {
                    "type": "hand_result",
                    "rewards": reward_dict,
                    "showdown": showdown,
                    "players": final_players,
                    "table_cards": community,
                }
            )

            self._wait_for_hand_ack()

        final_stacks = {a.name: float(a.stack) for a in agents}
        self._broadcast_sync(
            {
                "type": "game_over",
                "final_stacks": final_stacks,
            }
        )

    # ------------------------------------------------------------------
    # Push helpers
    # ------------------------------------------------------------------

    def _push_your_turn(self, seat: int, obs: Observation):
        """Send your_turn payload only to the acting player."""
        player = self.players.get(seat)
        if player is None or not player.is_connected:
            return
        payload = json.dumps(
            {
                "type": "your_turn",
                "observation": _observation_to_dict(obs),
            }
        )
        asyncio.run_coroutine_threadsafe(
            _safe_send(player._websocket, payload), self._loop
        )

    def _broadcast_player_states(self, agents):
        """After reset_hand, push each player their own hole cards and stack."""

        for seat, player in self.players.items():
            if not player.is_connected or seat >= len(agents):
                continue
            agent = agents[seat]
            hand = []
            for c in agent.cards:
                suit_bitmask = TreysCard.get_suit_int(c)
                rank_idx = TreysCard.get_rank_int(c)
                suit_idx = int(math.log2(suit_bitmask)) if suit_bitmask > 0 else 0
                hand.append({"suit": suit_idx, "rank": rank_idx})
            payload = json.dumps(
                {
                    "type": "player_state",
                    "stack": float(agent.stack),
                    "hand_cards": hand,
                }
            )
            asyncio.run_coroutine_threadsafe(
                _safe_send(player._websocket, payload), self._loop
            )

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    def _broadcast_sync(self, payload: dict):
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_async(json.dumps(payload)), self._loop
        )

    async def _broadcast_async(self, raw: str):
        for player in list(self.players.values()):
            if player.is_connected:
                try:
                    await player._websocket.send(raw)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def _start_http_server(self):
        html = _HTML_TEMPLATE.replace("{{WS_PORT}}", str(self.ws_port))
        _HTTPHandler.html = html

        def _run():
            server = HTTPServer((self.host, self.http_port), _HTTPHandler)
            log.info("HTTP server on port %d", self.http_port)
            server.serve_forever()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.3)

    def _wait_for_hand_ack(self):
        human_seats = [
            seat
            for seat, player in self.players.items()
            if not player.is_ai and player.is_connected
        ]
        if not human_seats:
            return

        events = {}
        for seat in human_seats:
            e = threading.Event()
            self._hand_ack_events[seat] = e
            events[seat] = e

        for e in events.values():
            e.wait(timeout=60)

        for seat in human_seats:
            self._hand_ack_events.pop(seat, None)


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------


def _build_table_update(obs: Observation, agents=None) -> dict:
    """
    Build a table_update payload.
    If agents is provided, includes all_players with real names and seat numbers.
    """
    payload = {
        "type": "table_update",
        "street": int(obs.street),
        "pot": float(obs.pot),
        "bet_to_match": float(obs.bet_to_match),
        "table_cards": [
            {"suit": int(c.suit), "rank": int(c.rank)}
            for c in obs.table_cards.cards
            if int(c.rank) > 0
        ],
        # Legacy field kept for compatibility
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
    if agents is not None:
        payload["all_players"] = [
            {
                "seat": i,
                "name": a.name,
                "position": int(a.position),
                "state": a.state.value,
                "stack": float(a.stack),
                "money_in_pot": float(a.money_in_pot),
                "bet_this_street": float(a.bet_this_street),
                "is_all_in": bool(a.all_in),
            }
            for i, a in enumerate(agents)
        ]
    return payload


def _build_turn_indicator(obs: Observation, agents) -> dict:
    """Let all clients know whose turn it is (for UI highlighting)."""
    seat = int(obs.player_identifier)
    name = agents[seat].name if seat < len(agents) else "?"
    return {"type": "turn_indicator", "seat": seat, "name": name}


async def _safe_send(websocket, payload: str):
    try:
        await websocket.send(payload)
    except Exception:
        pass
