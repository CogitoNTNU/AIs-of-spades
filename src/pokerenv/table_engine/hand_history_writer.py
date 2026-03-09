# table_engine/hand_history_writer.py
import time
import numpy as np
from treys import Card
from pokerenv.common import GameState, PlayerState
from pokerenv.utils import pretty_print_hand

SB = 2.5
BB = 5


class HandHistoryWriter:
    """
    Handles all hand history formatting and writing to disk.
    Can be enabled/disabled without affecting game logic.
    """

    def __init__(
        self,
        location: str = "hands/",
        enabled: bool = False,
        track_single_player: bool = False,
    ):
        self.location = location
        self.enabled = enabled
        self.track_single_player = track_single_player
        self.history = []

    def reset(self):
        self.history = []

    def write(self, text: str):
        if self.enabled:
            self.history.append(text)

    def initialize(self, players: list):
        if not self.enabled:
            return
        t = time.localtime()
        self.history.append(
            "PokerStars Hand #%d: Hold'em No Limit ($%.2f/$%.2f USD) - %d/%d/%d %d:%d:%d ET"
            % (
                np.random.randint(2230397, 32303976),
                SB,
                BB,
                t.tm_year,
                t.tm_mon,
                t.tm_mday,
                t.tm_hour,
                t.tm_min,
                t.tm_sec,
            )
        )
        self.history.append("Table 'Wempe III' 6-max Seat #2 is the button")
        for i, player in enumerate(players):
            self.history.append(
                "Seat %d: %s ($%.2f in chips)" % (i + 1, player.name, player.stack * BB)
            )

    def write_hole_cards(self, players: list):
        if not self.enabled:
            return
        self.history.append("*** HOLE CARDS ***")
        for player in players:
            if self.track_single_player or player.identifier == 0:
                self.history.append(
                    "Dealt to %s [%s %s]"
                    % (
                        player.name,
                        Card.int_to_str(player.cards[0]),
                        Card.int_to_str(player.cards[1]),
                    )
                )

    def write_showdown(self, players: list, evaluator, community_cards: list):
        if not self.enabled:
            return
        self.history.append("*** SHOW DOWN ***")
        active_players = [p for p in players if p.state is PlayerState.ACTIVE]
        hand_types = [
            evaluator.class_to_string(evaluator.get_rank_class(p.hand_rank))
            for p in active_players
        ]
        for player in active_players:
            player_hand_type = evaluator.class_to_string(
                evaluator.get_rank_class(player.hand_rank)
            )
            matches = len([m for m in hand_types if m == player_hand_type])
            multiple = matches > 1
            self.history.append(
                "%s: shows [%s %s] (%s)"
                % (
                    player.name,
                    Card.int_to_str(player.cards[0]),
                    Card.int_to_str(player.cards[1]),
                    pretty_print_hand(
                        player.cards, player_hand_type, community_cards, multiple
                    ),
                )
            )

    def write_summary(
        self, players: list, pot: float, street: GameState, community_cards: list
    ):
        if not self.enabled:
            return
        for player in players:
            if player.winnings_for_hh > 0:
                self.history.append(
                    "%s collected $%.2f from pot"
                    % (player.name, player.winnings_for_hh * BB)
                )
        self.history.append("*** SUMMARY ***")
        self.history.append("Total pot $%.2f | Rake $%.2f" % (pot * BB, 0))

        c = community_cards
        if street == GameState.FLOP and len(c) >= 3:
            self.history.append(
                "Board [%s %s %s]"
                % (
                    Card.int_to_str(c[0]),
                    Card.int_to_str(c[1]),
                    Card.int_to_str(c[2]),
                )
            )
        elif street == GameState.TURN and len(c) >= 4:
            self.history.append(
                "Board [%s %s %s %s]"
                % (
                    Card.int_to_str(c[0]),
                    Card.int_to_str(c[1]),
                    Card.int_to_str(c[2]),
                    Card.int_to_str(c[3]),
                )
            )
        elif street == GameState.RIVER and len(c) >= 5:
            self.history.append(
                "Board [%s %s %s %s %s]"
                % (
                    Card.int_to_str(c[0]),
                    Card.int_to_str(c[1]),
                    Card.int_to_str(c[2]),
                    Card.int_to_str(c[3]),
                    Card.int_to_str(c[4]),
                )
            )

    def flush_to_disk(self):
        if not self.enabled or self.location is None or not self.history:
            return
        filepath = "%shandhistory_%s.txt" % (self.location, time.time())
        with open(filepath, "w") as f:
            for row in self.history:
                f.writelines(row + "\n")
