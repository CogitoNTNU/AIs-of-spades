# test_table.py
from pokerenv.table import Table
from test_player import TestPlayerAgent
from pokerenv.observation import Observation
from pokerenv.common import GameState
from pokerenv.action import Action
from treys import Card


STREET_NAMES = {
    GameState.PREFLOP: "PREFLOP",
    GameState.FLOP: "FLOP",
    GameState.TURN: "TURN",
    GameState.RIVER: "RIVER",
}


class TestTable:
    """
    Wrapper around Table for interactive testing.
    - Creates N TestPlayerAgents automatically
    - Exposes step(action), fold(), call(), bet(amount)
    - snapshot() returns full serializable table state for UIs
    - print_snapshot() prints state to stdout
    """

    def __init__(self, n_players: int = 3, stack_low: int = 50, stack_high: int = 200):
        self.n_players = n_players
        self.players = [
            TestPlayerAgent(i, "Player_%d" % (i + 1)) for i in range(n_players)
        ]
        self.table = Table(
            n_players=n_players,
            players=self.players,
            stack_low=stack_low,
            stack_high=stack_high,
            hand_history_location=None,
            invalid_action_penalty=0,
        )
        self.table.hh.enabled = True
        self.hand_log = []
        self.current_obs = None
        self.done = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        self.hand_log = []
        self.done = False
        obs_array = self.table.reset_hand()
        self.current_obs = Observation(obs_array)
        return self.snapshot()

    def step(self, action: Action) -> dict:
        if self.done:
            raise RuntimeError("Hand is over — call reset() to start a new hand.")

        acting_i = int(self.current_obs.player_identifier)
        self.players[acting_i].set_next_action(action)

        obs_array, rewards, done, _ = self.table.step(action)
        self.done = done
        self.current_obs = Observation(obs_array) if not done else None

        # Sync hand log with table's internal history
        self.hand_log = list(self.table.hh.history)

        return self.snapshot()

    # Convenience shortcuts
    def fold(self) -> dict:
        return self.step(self.players[self._acting_i()].make_fold())

    def call(self) -> dict:
        return self.step(self.players[self._acting_i()].make_call())

    def bet(self, amount: float) -> dict:
        return self.step(self.players[self._acting_i()].make_bet(amount))

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        t = self.table
        community = [Card.int_to_str(c) for c in t.street_mgr.cards]

        players_info = []
        for p in self.players:
            players_info.append(
                {
                    "identifier": p.identifier,
                    "name": p.name,
                    "position": p.position,
                    "stack": round(float(p.stack), 2),
                    "money_in_pot": round(float(p.money_in_pot), 2),
                    "bet_this_street": round(float(p.bet_this_street), 2),
                    "state": p.state.name,
                    "all_in": p.all_in,
                    "cards": [Card.int_to_str(c) for c in p.cards] if p.cards else [],
                    "winnings": round(float(p.winnings), 2),
                }
            )

        acting_player = None
        valid_actions = None
        if not self.done and self.current_obs is not None:
            acting_i = self._acting_i()
            acting_player = acting_i
            va = t.betting.get_valid_actions(self.players[acting_i], self.players)
            valid_actions = {
                "actions": [a.name for a in va["actions_list"]],
                "bet_range": [
                    round(va["bet_range"][0], 2),
                    round(va["bet_range"][1], 2),
                ],
            }

        return {
            "street": STREET_NAMES.get(t.street_mgr.street, "UNKNOWN"),
            "pot": round(float(t.pot_mgr.pot), 2),
            "bet_to_match": round(float(t.betting.bet_to_match), 2),
            "minimum_raise": round(float(t.betting.minimum_raise), 2),
            "community_cards": community,
            "players": players_info,
            "acting_player": acting_player,
            "valid_actions": valid_actions,
            "done": self.done,
            "hand_log": list(self.hand_log),
        }

    def print_snapshot(self):
        s = self.snapshot()
        print("\n" + "═" * 56)
        print(
            "  %-10s │ pot: %-6.2f │ to match: %-6.2f │ min raise: %.2f"
            % (s["street"], s["pot"], s["bet_to_match"], s["minimum_raise"])
        )
        board = "  Board: %s" % (
            " ".join(s["community_cards"]) if s["community_cards"] else "(none yet)"
        )
        print(board)
        print("─" * 56)
        for p in s["players"]:
            cards = " ".join(p["cards"]) if p["cards"] else "?? ??"
            marker = " ◄ TO ACT" if p["identifier"] == s["acting_player"] else ""
            print(
                "  [%d] %-12s %s  stack:%-6.1f in_pot:%-5.1f %-8s%s"
                % (
                    p["identifier"],
                    p["name"],
                    cards,
                    p["stack"],
                    p["money_in_pot"],
                    p["state"],
                    marker,
                )
            )
        if s["valid_actions"] and not s["done"]:
            print("─" * 56)
            print(
                "  Valid: %-30s  bet: [%.1f – %.1f]"
                % (
                    ", ".join(s["valid_actions"]["actions"]),
                    s["valid_actions"]["bet_range"][0],
                    s["valid_actions"]["bet_range"][1],
                )
            )
        if s["done"]:
            print("─" * 56)
            print("  ★  HAND OVER")
            for p in s["players"]:
                delta = p["winnings"]
                sign = "+" if delta >= 0 else ""
                print("     %-14s  %s%.2f BB" % (p["name"], sign, delta))
            print("\n  Hand log:")
            for line in s["hand_log"]:
                print("    " + line)
        print("═" * 56 + "\n")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _acting_i(self) -> int:
        if self.current_obs is None:
            raise RuntimeError("No active observation — hand may be over.")
        return int(self.current_obs.player_identifier)
