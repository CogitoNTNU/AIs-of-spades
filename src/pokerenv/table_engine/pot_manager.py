from pokerenv.common import PlayerState

BB = 5


class PotManager:
    def __init__(self):
        self.pot = 0.0
        self.total_pot_for_hh = 0.0

    def reset(self):
        self.pot = 0.0
        self.total_pot_for_hh = 0.0

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
            last_bet_placed_by.total_invested -= amount
            history_writer(
                "Uncalled bet ($%.2f) returned to %s"
                % (amount * BB, last_bet_placed_by.name)
            )

    def distribute_with_cards(self, players: list, evaluator, community_cards: list):
        active_players = [p for p in players if p.state is PlayerState.ACTIVE]

        # Step 1 — hand ranks
        for player in active_players:
            player.calculate_hand_rank(evaluator, community_cards)

        # self.pot already contains all contributions including folded players,
        # added incrementally via pot_mgr.add() during betting.
        self.total_pot_for_hh = self.pot

        # Step 2 — uncontested pot
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.pot
            winner.winnings += self.pot
            winner.winnings_for_hh += self.pot
            self.pot = 0.0
            for player in players:
                player.money_in_pot = 0
            return

        # Step 3 — side pot loop
        # All players (including folded) contribute to pot sizing,
        # but only active players are eligible to win.
        # Sort ascending by money_in_pot to peel side pots from the bottom up.
        all_contributors = sorted(players, key=lambda p: p.money_in_pot)

        while any(p.money_in_pot > 0 for p in all_contributors):
            # Smallest contribution among players who still have chips in the pot
            min_contribution = min(
                p.money_in_pot for p in all_contributors if p.money_in_pot > 0
            )

            # Side pot is the sum of each player's contribution capped at min_contribution
            side_pot = sum(
                min(p.money_in_pot, min_contribution) for p in all_contributors
            )

            for player in all_contributors:
                player.money_in_pot = max(player.money_in_pot - min_contribution, 0)

            # Only active players are eligible to win this side pot
            eligible = [p for p in all_contributors if p.state is PlayerState.ACTIVE]

            if eligible:
                best_rank = min(p.hand_rank for p in eligible)
                winners = [p for p in eligible if p.hand_rank == best_rank]
                share = side_pot / len(winners)
                for winner in winners:
                    winner.stack += share
                    winner.winnings += share
                    winner.winnings_for_hh += share
            else:
                # Degenerate case: no active players eligible for this side pot.
                # Chips are consumed to prevent an infinite loop — should never happen
                # in a well-formed game where at least one active player remains.
                pass

            # Drop players who have exhausted their pot contribution
            all_contributors = [p for p in all_contributors if p.money_in_pot > 0]

        self.pot = 0.0
