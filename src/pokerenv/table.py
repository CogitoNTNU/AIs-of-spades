# table.py
import numpy as np
import math
import gymnasium as gym
from treys import Evaluator, Card
from pokerenv.common import PlayerState, PlayerAction, TablePosition
from pokerenv.action import Action
from pokerenv.player import Player
from pokerenv.table_engine import (
    BettingManager,
    PotManager,
    StreetManager,
    HandHistoryWriter,
)
from pokerenv.utils import approx_gt, approx_lte

BB = 5
MIN_STACK_TO_PLAY = 1


class Table(gym.Env):
    def __init__(
        self,
        n_players,
        players,
        track_single_player=False,
        stack_low=50,
        stack_high=200,
        hand_history_location="hands/",
        hand_history_enabled=False,
        invalid_action_penalty=0,
        evaluator=None,
    ):
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(4), gym.spaces.Box(-math.inf, math.inf, (1, 1)))
        )
        self.observation_space = gym.spaces.Box(-math.inf, math.inf, (58, 1))

        self.n_players = n_players
        self.players: list[Player] = players
        self.stack_low = stack_low
        self.stack_high = stack_high

        self.evaluator = evaluator if evaluator is not None else Evaluator()
        self.rng = np.random.default_rng(None)

        self.betting = BettingManager()
        self.pot_mgr = PotManager()
        self.street_mgr = StreetManager()
        self.hh = HandHistoryWriter(
            location=hand_history_location,
            enabled=hand_history_enabled,
            track_single_player=track_single_player,
        )

        self.current_turn = 0
        self.current_player_i = 0
        self.next_player_i = min(self.n_players - 1, 2)
        self.active_players = n_players
        self.street_finished = False
        self.hand_is_over = False
        self.first_to_act = None

        self.hand_number = 0
        self.hand_log_shape = (64, 8)
        self.hand_log = np.full(self.hand_log_shape, -1.0)

        self.dealer_position = 0
        self.showdown_cards: dict = {}

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.street_mgr.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _start_new_hand(self, reset_stacks=False):
        # Reset core state
        self.current_turn = 0
        self.first_to_act = None
        self.street_finished = False
        self.hand_is_over = False

        self.hand_log = np.full(self.hand_log_shape, -1.0)
        self.hand_number = 0

        # Reset managers
        self.betting.reset()
        self.pot_mgr.reset()
        self.street_mgr.reset()
        self.hh.reset()

        # Assign positions based on dealer
        for i, player in enumerate(self.players):
            player.position = (i - self.dealer_position) % self.n_players

        # Deal cards
        initial_draw = self.street_mgr.deal_hole_cards(self.n_players)

        # Reset players
        self.active_players = 0
        for i, player in enumerate(self.players):
            player.reset()

            # Optionally reset stacks (only in reset())
            if reset_stacks:
                player.set_stack(
                    self.rng.integers(self.stack_low, self.stack_high, 1)[0]
                )

            # Determine state
            if player.stack < MIN_STACK_TO_PLAY:
                player.stack = 0
                player.state = PlayerState.OUT
                continue

            player.state = PlayerState.ACTIVE
            self.active_players += 1

            # Assign cards
            player.cards = [
                initial_draw[i],
                initial_draw[i + self.n_players],
            ]

        if self.active_players == 0:
            raise Exception("No active players")

        # Hand history init
        if self.hh.enabled:
            self.hh.initialize(self.players)

        # Post blinds
        self._post_blinds()

        # Reset action flags
        for player in self.players:
            player.acted_this_street = False

        # Determine first player to act
        active_indices = [
            i for i, p in enumerate(self.players) if p.state is PlayerState.ACTIVE
        ]

        if self.n_players == 2:
            first_i = next(
                (
                    i
                    for i in active_indices
                    if self.players[i].position == TablePosition.SB
                ),
                active_indices[0],
            )
        else:
            first_i = next(
                (i for i in active_indices if self.players[i].position == 2),  # UTG
                active_indices[0],
            )

        self.next_player_i = first_i
        self.current_player_i = first_i

        # Write hole cards
        if self.hh.enabled:
            self.hh.write_hole_cards(self.players)

        return self._get_observation(self.players[self.next_player_i])

    def reset(self):
        # New episode → reset dealer + stacks
        self.dealer_position = 0
        return self._start_new_hand(reset_stacks=True)

    def reset_hand(self):
        # Same episode → rotate dealer, keep stacks
        self.dealer_position = (self.dealer_position + 1) % self.n_players
        return self._start_new_hand(reset_stacks=False)

    def step(self, action: Action):
        self.current_player_i = self.next_player_i
        player = self.players[self.current_player_i]
        self.current_turn += 1

        if (
            player.state is not PlayerState.ACTIVE
        ) and not self.hand_is_over:
            raise Exception("A player who is inactive or all-in was allowed to act")

        if self.first_to_act is None:
            self.first_to_act = player

        if not self.hand_is_over:
            self._apply_action(player, action)
            self.hand_number += 1
            self._check_street_or_hand_over()

        if self.hand_is_over:
            self._end_hand()
            obs = np.zeros(58, dtype=np.float32)
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
        active_players = sorted(
            [p for p in self.players if p.state in (PlayerState.ACTIVE, PlayerState.ALL_IN)],
            key=lambda p: p.position,
        )

        if len(active_players) < 2:
            return

        if len(active_players) == 2:
            # Dealer is SB in heads-up
            sb_player = self._get_next_active_by_position(TablePosition.SB)
            bb_player = [p for p in active_players if p != sb_player][0]
        else:
            sb_player = self._get_next_active_by_position(TablePosition.SB)

            sb_index = active_players.index(sb_player)
            bb_player = active_players[(sb_index + 1) % len(active_players)]

        # Small blind
        amount = min(0.5, sb_player.stack)
        if amount > 0:
            self.pot_mgr.add(sb_player.bet(amount))
            self.betting.change_bet_to_match(amount)
            self.hh.write("%s: posts small blind $%.2f" % (sb_player.name, amount * BB))

        # Big blind
        amount = min(1, bb_player.stack)
        if amount > 0:
            self.pot_mgr.add(bb_player.bet(amount))
            self.betting.change_bet_to_match(amount)
            self.betting.minimum_raise = amount
            self.betting.last_bet_placed_by = bb_player
            self.hh.write("%s: posts big blind $%.2f" % (bb_player.name, amount * BB))

    def _apply_action(self, player, action: Action):
        valid_actions = self.betting.get_valid_actions(player, self.players)
        action_list = valid_actions["actions_list"]
        bet_range = valid_actions["bet_range"]

        # If the requested action is unavailable → FOLD, except BET → CALL
        if action.action_type not in action_list:
            fallback = PlayerAction.CALL if action.action_type is PlayerAction.BET else PlayerAction.FOLD
            action = Action(
                action_type=fallback,
                observation=action.observation,
                bet_amount=0.0,
                bet_normalized=action.bet_normalized,
            )
        elif action.action_type is PlayerAction.BET:
            clamped = float(np.clip(action.bet_amount, bet_range[0], bet_range[1]))
            clamped = min(clamped, player.stack)
            action = Action(
                action_type=action.action_type,
                observation=action.observation,
                bet_amount=clamped,
                bet_normalized=action.bet_normalized,
            )
            out_of_range = not (
                approx_lte(bet_range[0], action.bet_amount)
                and approx_lte(action.bet_amount, bet_range[1])
            )
            if out_of_range or approx_gt(action.bet_amount, player.stack):
                action = Action(
                    action_type=PlayerAction.CALL,
                    observation=action.observation,
                    bet_amount=0.0,
                    bet_normalized=action.bet_normalized,
                )

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
                    (call_size / self.pot_mgr.pot) if self.pot_mgr.pot > 0 else 0.0,
                    self.street_mgr.street,
                )
                self.hh.write(
                    "%s: calls $%.2f and is all-in"
                    % (player.name, self.betting.bet_to_match * BB)
                )
            else:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.CALL.value,
                    (call_size / self.pot_mgr.pot) if self.pot_mgr.pot > 0 else 0.0,
                    self.street_mgr.street,
                )
                self.hh.write(
                    "%s: calls $%.2f" % (player.name, self.betting.bet_to_match * BB)
                )

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
                    (actual / self.pot_mgr.pot) if self.pot_mgr.pot > 0 else 0.0,
                    self.street_mgr.street,
                )
                self.hh.write("%s: bets $%.2f%s" % (player.name, actual * BB, suffix))
            else:
                self.update_hand_log(
                    player.identifier,
                    PlayerAction.BET.value,
                    (actual / self.pot_mgr.pot) if self.pot_mgr.pot > 0 else 0.0,
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

    def _check_street_or_hand_over(self):
        # Players who can still voluntarily act
        players_with_actions = [
            p for p in self.players if p.state is PlayerState.ACTIVE
        ]
        # Players who still owe action: haven't acted yet OR are behind on the bet
        players_who_should_act = [
            p
            for p in players_with_actions
            if not p.acted_this_street or p.bet_this_street != self.betting.bet_to_match
        ]

        if len(players_who_should_act) == 0:
            # Everyone who can act has acted and matched — street or hand is over.
            if self.active_players <= 1:
                self._return_uncalled_and_finish(hand_over=True)
            else:
                self._return_uncalled_and_finish(hand_over=False)
        else:
            self._advance_next_player()

    def _return_uncalled_and_finish(self, hand_over: bool):
        """Return any uncalled portion then end the hand or transition to next street."""
        all_active = [p for p in self.players if p.state in (PlayerState.ACTIVE, PlayerState.ALL_IN)]
        if self.betting.last_bet_placed_by is not None:
            biggest_other = max(
                (
                    p.bet_this_street
                    for p in all_active
                    if p is not self.betting.last_bet_placed_by
                ),
                default=0.0,
            )
            uncalled = max(
                self.betting.last_bet_placed_by.bet_this_street - biggest_other, 0.0
            )
            if uncalled > 0:
                self.pot_mgr.return_uncalled_bet(
                    self.betting.last_bet_placed_by, uncalled, self.hh.write
                )

        if hand_over:
            self.hand_is_over = True
        else:
            players_can_act = [
                p
                for p in self.players
                if p.state is PlayerState.ACTIVE
            ]
            self._do_street_transition(transition_to_end=(len(players_can_act) == 0))

    def _advance_next_player(self):
        current_pos = self.players[self.current_player_i].position

        # Candidates: active, not all-in, different from current player
        candidates = [
            i
            for i in range(self.n_players)
            if self.players[i].state is PlayerState.ACTIVE
            and i != self.current_player_i
        ]

        if not candidates:
            # Current player is the only ACTIVE one left — no advancement needed;
            # _check_street_or_hand_over will close the hand/street on the next
            # check after the player has acted.
            self.next_player_i = self.current_player_i
            return

        after = [i for i in candidates if self.players[i].position > current_pos]
        before = [i for i in candidates if self.players[i].position <= current_pos]

        self.next_player_i = (
            min(after, key=lambda i: self.players[i].position)
            if after
            else min(before, key=lambda i: self.players[i].position)
        )

    def _do_street_transition(self, transition_to_end=False):
        active_can_act = [
            p for p in self.players if p.state is PlayerState.ACTIVE
        ]
        if len(active_can_act) <= 1:
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
        self.showdown_cards = {
            p.name: p.cards
            for p in self.players
            if p.state is not PlayerState.FOLDED and p.state is not PlayerState.OUT
        }

        while len(self.street_mgr.cards) < 5:
            remaining = self.street_mgr.transition(
                self.players, self.hh.write, transition_to_end=True
            )
            if remaining:
                break

        self.pot_mgr.distribute_with_cards(
            self.players, self.evaluator, self.street_mgr.cards
        )
        self.hh.write_showdown(self.players, self.evaluator, self.street_mgr.cards)
        self.hh.write_summary(
            self.players,
            self.pot_mgr.total_pot_for_hh,  # ← was pot_mgr.pot (always 0 after distribute)
            self.street_mgr.street,
            self.street_mgr.cards,
        )
        self.hh.flush_to_disk()

    def _get_next_active_by_position(self, target_pos):
        candidates = [p for p in self.players if p.state in (PlayerState.ACTIVE, PlayerState.ALL_IN)]
        return min(candidates, key=lambda p: (p.position - target_pos) % self.n_players)

    def _get_observation(self, player):
        observation = np.zeros(58, dtype=np.float32)
        observation[0] = player.identifier

        valid_actions = self.betting.get_valid_actions(player, self.players)
        for action in valid_actions["actions_list"]:
            observation[action.value + 1] = 1
        observation[4] = valid_actions["bet_range"][0]
        observation[5] = valid_actions["bet_range"][1]

        observation[6] = player.position
        observation[7] = Card.get_suit_int(player.cards[0])
        observation[8] = Card.get_rank_int(player.cards[0])
        observation[9] = Card.get_suit_int(player.cards[1])
        observation[10] = Card.get_rank_int(player.cards[1])
        observation[11] = player.stack
        observation[12] = player.money_in_pot
        observation[13] = player.bet_this_street

        observation[14] = self.street_mgr.street
        for i, card in enumerate(self.street_mgr.cards):
            observation[15 + (i * 2)] = Card.get_suit_int(card)
            observation[16 + (i * 2)] = Card.get_rank_int(card)
        observation[25] = self.pot_mgr.pot
        observation[26] = self.betting.bet_to_match
        observation[27] = self.betting.minimum_raise

        others = [other for other in self.players if other is not player]
        for i, other in enumerate(others):
            observation[28 + i * 6] = other.position
            observation[29 + i * 6] = other.state.value
            observation[30 + i * 6] = other.stack
            observation[31 + i * 6] = other.money_in_pot
            observation[32 + i * 6] = other.bet_this_street
            observation[33 + i * 6] = int(other.all_in)

        return observation

    def update_hand_log(self, player_id, action_value, bet_fraction, street):
        hand_log_line = np.zeros(self.hand_log_shape[-1])

        stack_norm = self.players[player_id].stack / self.stack_high
        pot_norm = self.pot_mgr.pot / (
            self.stack_high * self.n_players
        )  # reasonable scale
        active = sum(1 for p in self.players if p.state in (PlayerState.ACTIVE, PlayerState.ALL_IN))
        btm_norm = self.betting.bet_to_match / self.stack_high

        hand_log_line[:8] = [
            player_id,
            action_value,
            bet_fraction,
            street,
            pot_norm,
            btm_norm,
            active,
            stack_norm,
        ]

        if self.hand_number < self.hand_log.shape[0]:
            self.hand_log[self.hand_number] = hand_log_line
        else:
            self.hand_log = np.roll(self.hand_log, -1, axis=0)
            self.hand_log[-1] = hand_log_line
