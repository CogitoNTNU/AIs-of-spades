# table_engine/pot_manager.py
from pokerenv.common import PlayerState

BB = 5


class PotManager:
    """
    Handles pot accumulation and distribution at showdown,
    including side pot logic for all-in players.
    """

    def __init__(self):
        self.pot = 0.0

    def reset(self):
        self.pot = 0.0

    def add(self, amount: float):
        self.pot += amount

    def subtract(self, amount: float):
        self.pot -= amount

    def return_uncalled_bet(self, last_bet_placed_by, amount: float, history_writer):
        if amount > 0 and last_bet_placed_by is not None:
            self.pot -= amount
            last_bet_placed_by.stack += amount
            last_bet_placed_by.money_in_pot -= amount
            last_bet_placed_by.bet_this_street -= amount
            history_writer(
                "Uncalled bet ($%.2f) returned to %s"
                % (amount * BB, last_bet_placed_by.name)
            )

    def distribute_with_cards(self, players: list, evaluator, community_cards: list):
        """
        Distributes the pot among active players after computing hand ranks.
        Handles side pots for all-in situations correctly.
        """
        # Calculate hand ranks for all active players
        active_players = [p for p in players if p.state is PlayerState.ACTIVE]
        for player in active_players:
            player.calculate_hand_rank(evaluator, community_cards)

        # Collect folded players' contributions and zero out their money_in_pot
        # immediately to avoid double-subtraction in the side pot loop below
        for player in players:
            if player.state is not PlayerState.ACTIVE:
                self.pot += player.money_in_pot
                player.winnings -= player.money_in_pot
                player.money_in_pot = 0

        # Early exit: only one active player remaining
        if len(active_players) == 1:
            winner = active_players[0]
            winner.winnings += self.pot + winner.money_in_pot
            winner.winnings_for_hh += self.pot + winner.money_in_pot
            winner.money_in_pot = 0
            return

        # Side pot loop: peel off one layer at a time (handles all-in players)
        remaining = list(active_players)
        pot = 0.0

        while remaining:
            min_money_in_pot = min(p.money_in_pot for p in remaining)

            for player in remaining:
                pot += min_money_in_pot
                player.money_in_pot -= min_money_in_pot
                player.winnings -= min_money_in_pot

            best_rank = min(p.hand_rank for p in remaining)
            winners = [p for p in remaining if p.hand_rank == best_rank]
            share = pot / len(winners)
            for winner in winners:
                winner.winnings += share
                winner.winnings_for_hh += share

            remaining = [p for p in remaining if p.money_in_pot > 0]
            pot = 0.0

        # Return leftover money_in_pot to any player with uncollected chips
        for player in active_players:
            if player.money_in_pot > 0:
                player.winnings += player.money_in_pot
                player.winnings_for_hh += player.money_in_pot
                player.money_in_pot = 0
