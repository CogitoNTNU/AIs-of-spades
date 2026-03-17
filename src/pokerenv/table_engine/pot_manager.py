# table_engine/pot_manager.py
from pokerenv.common import PlayerState

BB = 5


class PotManager:
    """
    Handles pot accumulation and distribution at showdown,
    including side pot logic for all-in players.

    Reward semantics:
        player.winnings represents the NET change for the hand.
        Chips already paid into the pot are NOT deducted here —
        they were already subtracted from player.stack when bet/called.
        So winnings starts at 0 and only goes UP for winners.
        Folded players keep winnings = 0 (their loss is already in the stack).
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

        Steps:
        1. Compute hand ranks for all active (non-folded) players.
        2. Absorb folded players' money_in_pot into the pot — their
            winnings stay at 0 (loss already reflected in stack).
        3. If only one active player remains, they win the whole pot.
        4. Otherwise, peel side pots one layer at a time ordered by
            money_in_pot (smallest first), awarding each layer to the
            best hand among eligible players.

        Stack update happens here — get_reward() is pure and does not
        touch player.stack.
        """
        active_players = [p for p in players if p.state is PlayerState.ACTIVE]

        # Step 1 — hand ranks
        for player in active_players:
            player.calculate_hand_rank(evaluator, community_cards)

        # Step 2 — absorb folded/out contributions into pot
        for player in players:
            if player.state is not PlayerState.ACTIVE:
                self.pot += player.money_in_pot
                player.money_in_pot = 0

        # Step 3 — uncontested pot
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.pot
            winner.winnings += self.pot
            winner.winnings_for_hh += self.pot
            self.pot = 0.0
            winner.money_in_pot = 0
            return

        # Step 4 — side pot loop
        remaining = sorted(active_players, key=lambda p: p.money_in_pot)

        while remaining:
            min_contribution = min(p.money_in_pot for p in remaining)
            side_pot = min_contribution * len(remaining)

            for player in remaining:
                player.money_in_pot -= min_contribution

            best_rank = min(p.hand_rank for p in remaining)
            winners = [p for p in remaining if p.hand_rank == best_rank]
            share = side_pot / len(winners)
            for winner in winners:
                winner.stack += share
                winner.winnings += share
                winner.winnings_for_hh += share

            remaining = [p for p in remaining if p.money_in_pot > 0]

        self.pot = 0.0

        for player in players:
            player.money_in_pot = 0
