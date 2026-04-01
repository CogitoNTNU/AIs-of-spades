# table_engine/street_manager.py
from treys import Deck, Card
from pokerenv.common import GameState, PlayerState
import numpy as np


class StreetManager:
    """
    Handles street progression (PREFLOP → FLOP → TURN → RIVER → SHOWDOWN),
    deck management, and community cards.
    """

    def __init__(self):
        self.deck = Deck()
        self.cards = []
        self.street = GameState.PREFLOP
        self.rng = np.random.default_rng(None)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.deck.cards = Deck.GetFullDeck()
        self.rng.shuffle(self.deck.cards)
        self.cards = []
        self.street = GameState.PREFLOP

    def deal_hole_cards(self, n_players: int) -> list:
        """Draws and returns 2*n_players cards for hole card distribution."""
        return self.deck.draw(n_players * 2)

    def transition(
        self, players: list, history_writer, transition_to_end=False
    ) -> bool:
        """
        Advances the street. Returns True if the hand is now over (reached showdown).
        Resets per-street player state.
        """
        transitioned = False
        hand_is_over = False

        if self.street == GameState.PREFLOP:
            self.cards = self.deck.draw(3)
            history_writer(
                "*** FLOP *** [%s %s %s]"
                % (
                    Card.int_to_str(self.cards[0]),
                    Card.int_to_str(self.cards[1]),
                    Card.int_to_str(self.cards[2]),
                )
            )
            self.street = GameState.FLOP
            transitioned = True

        if self.street == GameState.FLOP and (not transitioned or transition_to_end):
            new = self.deck.draw(1)
            self.cards = self.cards + new
            history_writer(
                "*** TURN *** [%s %s %s] [%s]"
                % (
                    Card.int_to_str(self.cards[0]),
                    Card.int_to_str(self.cards[1]),
                    Card.int_to_str(self.cards[2]),
                    Card.int_to_str(self.cards[3]),
                )
            )
            self.street = GameState.TURN
            transitioned = True

        if self.street == GameState.TURN and (not transitioned or transition_to_end):
            new = self.deck.draw(1)
            self.cards = self.cards + new
            history_writer(
                "*** RIVER *** [%s %s %s %s] [%s]"
                % (
                    Card.int_to_str(self.cards[0]),
                    Card.int_to_str(self.cards[1]),
                    Card.int_to_str(self.cards[2]),
                    Card.int_to_str(self.cards[3]),
                    Card.int_to_str(self.cards[4]),
                )
            )
            self.street = GameState.RIVER
            transitioned = True

        if self.street == GameState.RIVER and (not transitioned or transition_to_end):
            hand_is_over = True

        for player in players:
            player.finish_street()

        return hand_is_over

    def first_to_act_after_transition(
        self, players: list, n_players: int
    ) -> int | None:
        """Returns the seat index of the first active non-all-in player after a street
        transition.  Post-flop action starts with the lowest *position* value (SB = 0),
        not the lowest seat index."""
        candidates = [
            i
            for i in range(n_players)
            if players[i].state is PlayerState.ACTIVE
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda i: players[i].position)
