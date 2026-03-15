# table.py
import numpy as np
import math
import gymnasium as gym
from treys import Evaluator, Card
from pokerenv.common import PlayerState, PlayerAction, TablePosition
from pokerenv.action import Action
from pokerenv.observation import Observation
from pokerenv.table_engine import (
    BettingManager,
    PotManager,
    StreetManager,
    HandHistoryWriter,
)

BB = 5


class Table(gym.Env):
    def __init__(
        self,
        n_players,
        players,
        track_single_player=False,
        stack_low=50,
        stack_high=200,
        hand_history_location="hands/",
        invalid_action_penalty=0,
    ):
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(4), gym.spaces.Box(-math.inf, math.inf, (1, 1)))
        )
        self.observation_space = gym.spaces.Box(-math.inf, math.inf, (59, 1))

        self.n_players = n_players
        self.players = players
        self.stack_low = stack_low
        self.stack_high = stack_high

        self.evaluator = Evaluator()
        self.rng = np.random.default_rng(None)

        self.betting = BettingManager()
        self.pot_mgr = PotManager()
        self.street_mgr = StreetManager()
        self.hh = HandHistoryWriter(
            location=hand_history_location,
            enabled=False,
            track_single_player=track_single_player,
        )

        self.current_turn = 0
        self.current_player_i = 0
        self.next_player_i = min(self.n_players - 1, 2)
        self.active_players = n_players
        self.street_finished = False
        self.hand_is_over = False
        self.first_to_act = None

        # actions history
        self.hand_number = 0
        self.hand_log = np.full((32, 4), -1.0)

        self.dealer_position = 0

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.street_mgr.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        self.current_turn = 0
        self.active_players = self.n_players
        self.next_player_i = 0 if self.n_players == 2 else 2
        self.current_player_i = self.next_player_i
        self.first_to_act = None
        self.street_finished = False
        self.hand_is_over = False

        self.hand_number = 0
        self.hand_log = np.full((32, 4), -1.0)

        self.betting.reset()
        self.pot_mgr.reset()
        self.street_mgr.reset()
        self.hh.reset()

        initial_draw = self.street_mgr.deal_hole_cards(self.n_players)
        for i, player in enumerate(self.players):
            player.reset()
            player.position = i
            player.cards = [initial_draw[i], initial_draw[i + self.n_players]]
            player.stack = self.rng.integers(self.stack_low, self.stack_high, 1)[0]

        if self.hh.enabled:
            self.hh.initialize(self.players)

        self._post_blinds()

        # Reset acted_this_street after blinds so preflop action starts clean
        for player in self.players:
            player.acted_this_street = False

        if self.hh.enabled:
            self.hh.write_hole_cards(self.players)

        return self._get_observation(self.players[self.next_player_i])

    def reset_hand(self):
        """Resets only cards and pot"""
        self.current_turn = 0
        self.active_players = self.n_players
        self.next_player_i = 0 if self.n_players == 2 else 2
        self.current_player_i = self.next_player_i
        self.first_to_act = None
        self.street_finished = False
        self.hand_is_over = False

        self.betting.reset()
        self.pot_mgr.reset()
        self.street_mgr.reset()
        self.hh.reset()

        self.dealer_position = (self.dealer_position + 1) % self.n_players
        for player in self.players:
            player.position = (player.position - self.dealer_position) % self.n_players

        self.next_player_i = 0 if self.n_players == 2 else 2
        self.current_player_i = self.next_player_i

        initial_draw = self.street_mgr.deal_hole_cards(self.n_players)
        for i, player in enumerate(self.players):
            player.reset()
            player.cards = [initial_draw[i], initial_draw[i + self.n_players]]

        MIN_STACK_TO_PLAY = 1
        for player in self.players:
            if player.stack < MIN_STACK_TO_PLAY:
                player.state = PlayerState.OUT
                self.active_players -= 1

        if self.hh.enabled:
            self.hh.initialize(self.players)

        self._post_blinds()

        for player in self.players:
            player.acted_this_street = False

        if self.hh.enabled:
            self.hh.write_hole_cards(self.players)

        return self._get_observation(self.players[self.next_player_i])

    def step(self, action: Action):
        self.current_player_i = self.next_player_i
        player = self.players[self.current_player_i]
        self.current_turn += 1

        if (
            player.all_in or player.state is not PlayerState.ACTIVE
        ) and not self.hand_is_over:
            raise Exception("A player who is inactive or all-in was allowed to act")

        if self.first_to_act is None:
            self.first_to_act = player

        if not (self.hand_is_over or self.street_finished):
            self._apply_action(player, action)
            self.hand_number += 1
            self._check_street_or_hand_over()

        if self.street_finished and not self.hand_is_over:
            self._do_street_transition()

        # After transition, if all active players are all-in, end the hand immediately
        if not self.hand_is_over:
            active_can_act = [
                p
                for p in self.players
                if p.state is PlayerState.ACTIVE and not p.all_in
            ]
            if len(active_can_act) == 0 and self.active_players > 1:
                self._do_street_transition(transition_to_end=True)

        if self.hand_is_over:
            self._end_hand()
            obs = Observation.empty()
        else:
            obs = self._get_observation(self.players[self.next_player_i])

        rewards = np.asarray([p.get_reward() for p in sorted(self.players)])
        return obs, rewards, self.hand_is_over

    def get_player_by_name(self, name):
        for player in self.players:
            if player.name == name:
                return player
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_blinds(self):
        for player in self.players:
            if player.position == TablePosition.SB:
                self.pot_mgr.add(player.bet(0.5))
                self.betting.change_bet_to_match(0.5)
                self.hh.write("%s: posts small blind $%.2f" % (player.name, BB / 2))
            elif player.position == TablePosition.BB:
                self.pot_mgr.add(player.bet(1))
                self.betting.change_bet_to_match(1)
                self.betting.last_bet_placed_by = player
                self.hh.write("%s: posts big blind $%.2f" % (player.name, BB))

    def _apply_action(self, player, action: Action):
        valid_actions = self.betting.get_valid_actions(player, self.players)
        is_valid, fallback = self.betting.is_action_valid(player, action, valid_actions)

        if not is_valid:
            self._apply_fallback(player, fallback)
            return

        if action.action_type is PlayerAction.FOLD:
            player.fold()
            self.active_players -= 1
            self.update_hand_log(
                player.identifier, PlayerAction.FOLD.value, 0, self.street_mgr.street
            )
            self.hh.write("%s: folds" % player.name)

        elif action.action_type is PlayerAction.CALL:
            stack = player.stack
            call_size = player.check_or_call(self.betting.bet_to_match)
            self.pot_mgr.add(call_size)
            if self.betting.bet_to_match == 0 or call_size == 0:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.CALL.value,
                    0,
                    self.street_mgr.street,
                )
                self.hh.write("%s: checks" % player.name)
            elif player.all_in:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.CALL.value,
                    call_size / stack,
                    self.street_mgr.street,
                )
                self.hh.write(
                    "%s: calls $%.2f and is all-in" % (player.name, call_size * BB)
                )
            else:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.CALL.value,
                    call_size / stack,
                    self.street_mgr.street,
                )
                self.hh.write("%s: calls $%.2f" % (player.name, call_size * BB))

        elif action.action_type is PlayerAction.BET:
            stack = player.stack
            prev_bet = player.bet_this_street
            actual = player.bet(np.round(action.bet_amount, 2))
            self.pot_mgr.add(actual)
            total = actual + prev_bet
            suffix = " and is all-in" if player.all_in else ""
            if self.betting.bet_to_match == 0:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.BET.value,
                    actual / stack,
                    self.street_mgr.street,
                )
                self.hh.write("%s: bets $%.2f%s" % (player.name, actual * BB, suffix))
            else:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.BET.value,
                    actual / stack,
                    self.street_mgr.street,
                )
                self.hh.write(
                    "%s: raises $%.2f to $%.2f%s"
                    % (
                        player.name,
                        (total - self.betting.bet_to_match) * BB,
                        total * BB,
                        suffix,
                    )
                )
            self.betting.change_bet_to_match(total)
            self.betting.last_bet_placed_by = player

        else:
            raise Exception("Invalid action_type — use PlayerAction enum, not int")

    def _apply_fallback(self, player, fallback: PlayerAction):
        """Applies the fallback action chosen by BettingManager."""
        if fallback is PlayerAction.FOLD:
            player.fold()
            self.active_players -= 1
            self.update_hand_log(
                player.identifier, PlayerAction.FOLD.value, 0, self.street_mgr.street
            )
            self.hh.write("%s: folds" % player.name)
        elif fallback is PlayerAction.CALL:
            stack = player.stack
            call_size = player.check_or_call(self.betting.bet_to_match)
            self.pot_mgr.add(call_size)
            if self.betting.bet_to_match == 0 or call_size == 0:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.CALL.value,
                    0,
                    self.street_mgr.street,
                )
                self.hh.write("%s: checks" % player.name)
            elif player.all_in:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.CALL.value,
                    call_size / stack,
                    self.street_mgr.street,
                )
                self.hh.write(
                    "%s: calls $%.2f and is all-in" % (player.name, call_size * BB)
                )
            else:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.CALL.value,
                    call_size / stack,
                    self.street_mgr.street,
                )
                self.hh.write("%s: calls $%.2f" % (player.name, call_size * BB))

    def _check_street_or_hand_over(self):
        players_with_actions = [
            p for p in self.players if p.state is PlayerState.ACTIVE and not p.all_in
        ]
        players_who_should_act = [
            p
            for p in players_with_actions
            if not p.acted_this_street or p.bet_this_street != self.betting.bet_to_match
        ]

        if len(players_with_actions) < 2 and len(players_who_should_act) == 0:
            if self.active_players > 1:
                biggest_call = max(
                    (
                        p.bet_this_street
                        for p in self.players
                        if p.state is PlayerState.ACTIVE
                        and p is not self.betting.last_bet_placed_by
                    ),
                    default=0.0,
                )
                last_bet = (
                    self.betting.last_bet_placed_by.bet_this_street
                    if self.betting.last_bet_placed_by
                    else 0.0
                )
                uncalled = max(last_bet - biggest_call, 0.0)
                self.pot_mgr.return_uncalled_bet(
                    self.betting.last_bet_placed_by, uncalled, self.hh.write
                )
                self._do_street_transition(transition_to_end=True)
            else:
                uncalled = self.betting.minimum_raise
                self.pot_mgr.return_uncalled_bet(
                    self.betting.last_bet_placed_by, uncalled, self.hh.write
                )
                self.hand_is_over = True
        else:
            self._advance_next_player()

    def _advance_next_player(self):
        after = [
            i
            for i in range(self.n_players)
            if i > self.current_player_i
            and self.players[i].state is PlayerState.ACTIVE
            and not self.players[i].all_in
        ]
        before = [
            i
            for i in range(self.n_players)
            if i <= self.current_player_i
            and self.players[i].state is PlayerState.ACTIVE
            and not self.players[i].all_in
        ]
        self.next_player_i = min(after) if after else min(before)
        next_player = self.players[self.next_player_i]

        # Only mark street finished if the next player to act has already
        # matched the bet — i.e. they don't need to act again.
        # If they still owe chips (bet_this_street != bet_to_match) or
        # haven't acted yet, they must act first before the street ends.
        next_still_needs_to_act = (
            not next_player.acted_this_street
            or next_player.bet_this_street != self.betting.bet_to_match
        )

        if not next_still_needs_to_act:
            if self.betting.last_bet_placed_by is next_player or (
                self.first_to_act is next_player
                and self.betting.last_bet_placed_by is None
            ):
                self.street_finished = True
                if before:
                    self.next_player_i = min(before)

    def _do_street_transition(self, transition_to_end=False):
        active_can_act = [
            p for p in self.players if p.state is PlayerState.ACTIVE and not p.all_in
        ]
        if len(active_can_act) == 0:
            transition_to_end = True

        hand_over = self.street_mgr.transition(
            self.players, self.hh.write, transition_to_end=transition_to_end
        )
        self.betting.reset()
        self.first_to_act = None
        self.street_finished = False

        if hand_over:
            self.hand_is_over = True
        else:
            next_i = self.street_mgr.first_to_act_after_transition(
                self.players, self.n_players
            )
            if next_i is not None:
                self.next_player_i = next_i

    def _end_hand(self):
        # Safety net: ensure board is fully dealt before evaluation
        # (handles any edge case where _do_street_transition wasn't enough)
        while len(self.street_mgr.cards) < 5:
            remaining = self.street_mgr.transition(
                self.players, self.hh.write, transition_to_end=True
            )
            if remaining:
                break

        # 1. Distribute pot (also calculates hand ranks internally)
        self.pot_mgr.distribute_with_cards(
            self.players, self.evaluator, self.street_mgr.cards
        )
        # 2. Write showdown AFTER hand ranks are known
        self.hh.write_showdown(self.players, self.evaluator, self.street_mgr.cards)
        # 3. Write summary and flush to disk
        self.hh.write_summary(
            self.players,
            self.pot_mgr.pot,
            self.street_mgr.street,
            self.street_mgr.cards,
        )
        self.hh.flush_to_disk()

    def _get_observation(self, player):
        observation = np.zeros(59, dtype=np.float32)
        observation[0] = player.identifier

        valid_actions = self.betting.get_valid_actions(player, self.players)
        for action in valid_actions["actions_list"]:
            observation[action.value + 1] = 1
        observation[5] = valid_actions["bet_range"][0]
        observation[6] = valid_actions["bet_range"][1]

        observation[7] = player.position
        observation[8] = Card.get_suit_int(player.cards[0])
        observation[9] = Card.get_rank_int(player.cards[0])
        observation[10] = Card.get_suit_int(player.cards[1])
        observation[11] = Card.get_rank_int(player.cards[1])
        observation[12] = player.stack
        observation[13] = player.money_in_pot
        observation[14] = player.bet_this_street

        observation[15] = self.street_mgr.street
        for i, card in enumerate(self.street_mgr.cards):
            observation[16 + (i * 2)] = Card.get_suit_int(card)
            observation[17 + (i * 2)] = Card.get_rank_int(card)
        observation[26] = self.pot_mgr.pot
        observation[27] = self.betting.bet_to_match
        observation[28] = self.betting.minimum_raise

        others = [other for other in self.players if other is not player]
        for i, other in enumerate(others):
            observation[29 + i * 6] = other.position
            observation[30 + i * 6] = other.state.value
            observation[31 + i * 6] = other.stack
            observation[32 + i * 6] = other.money_in_pot
            observation[33 + i * 6] = other.bet_this_street
            observation[34 + i * 6] = int(other.all_in)

        return observation

    def update_hand_log(self, player_id, action_value, bet_fraction, street):
        if self.hand_number < self.hand_log.shape[0]:
            self.hand_log[self.hand_number] = [
                player_id,
                action_value,
                bet_fraction,
                street,
            ]
        else:
            self.hand_log = np.roll(self.hand_log, -1, axis=0)
            self.hand_log[-1] = [player_id, action_value, bet_fraction, street]
