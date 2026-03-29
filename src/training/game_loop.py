import time
import random as rn
import numpy as np

from pokerenv.table import Table
from pokerenv.observation import Observation
from pokerenv.common import PlayerState

from training.player_agent import PlayerAgent

MAIN_CHARACTER_NAME = "UGO"
MIN_STACK_TO_PLAY = 1


class Game:

    def __init__(self, opponents: list, current_model, config: dict):
        self.opponents = opponents
        self.current_model = current_model
        self.table = None
        self.agents = []
        self.config = config

        self.trajectory = []
        self.reward = 0.0

        self._elimination_bonus = float(config.get("elimination_bonus", 0.0))
        self._survival_bonus = float(config.get("survival_bonus", 0.0))
        self._isolation_bonus = float(config.get("isolation_bonus", 0.0))
        self._showdown_bonus = float(config.get("showdown_bonus", 0.0))
        self._steal_bonus = float(config.get("steal_bonus", 0.0))
        self._stack_lo = int(config.get("stack_lo", 50))
        self._stack_hi = int(config.get("stack_hi", 200))
        # Curriculum multiplier applied to main_reward at showdown.
        # Amplifies card-strength signal early in training; decays toward 1.0
        # as the agent matures and bluffing strategies become relevant.
        # LearningLoop computes the per-epoch value and injects it into game_config.
        self._showdown_reward_multiplier = float(
            config.get("showdown_reward_multiplier", 1.0)
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def reset(self):
        self.trajectory = []
        self.hand_rewards = []
        self.reward = 0.0

        self.active_opponents = rn.randint(1, 5)
        player_names = {i: f"player_{i + 1}" for i in range(6)}
        player_names[0] = MAIN_CHARACTER_NAME

        self.agents = [PlayerAgent(0, MAIN_CHARACTER_NAME, 0, self.current_model)]
        for n in range(1, self.active_opponents + 1):
            self.agents.append(
                PlayerAgent(n, player_names[n], 0, self.opponents[n - 1])
            )

        self.table = Table(
            self.active_opponents + 1,
            players=self.agents,
            stack_low=self._stack_lo,
            stack_high=self._stack_hi,
            hand_history_location="hands/",
            invalid_action_penalty=0,
        )
        self.table.seed(None)
        self.table.reset()

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def get_weighted_rewards(self, hand_rewards: list, gamma: float = 0.8) -> list:
        """
        Iterative discounted return computation across hands.
        Each hand's reward incorporates a discounted sum of future hands,
        so decisions early in the session receive credit for downstream outcomes.
        Iterative implementation avoids recursion overhead and list copying.
        """
        n = len(hand_rewards)
        if n == 0:
            return []
        result = list(hand_rewards)
        for i in range(n - 2, -1, -1):
            result[i] += gamma * result[i + 1]
        return result

    def _stack_snapshot(self) -> dict:
        """
        Capture opponent stacks before a hand starts.
        Used to detect eliminations by comparing against post-hand stacks.
        """
        return {i: self.agents[i].stack for i in range(1, len(self.agents))}

    def _compute_hand_bonus(
        self,
        stacks_before: dict,
        reached_showdown: bool,
        was_isolated: bool,
        was_steal: bool,
    ) -> tuple[float, dict]:
        """
        Compute pure outcome bonuses at the end of a hand.
        All bonuses are zero by default and must be enabled explicitly in config.
        None of them encode card strength or domain knowledge about hand values —
        they reward structural outcomes observable without knowing hole cards.

        elimination_bonus : fired for each opponent who had chips before the hand
                            and is now at zero. Rewards contributing to eliminations
                            without specifying how to achieve them.

        survival_bonus    : fired when the main character is the sole remaining
                            player with chips. Rare but a strong global success signal.

        isolation_bonus   : fired when the hand was played heads-up (1v1).
                            Rewards positional isolation play regardless of cards.

        showdown_bonus    : fired when the main character reached showdown.
                            Rewards consistent decision-making that sustains
                            involvement through the full hand, not card strength.

        steal_bonus       : fired when all opponents folded before showdown, the
                            main character was not all-in, and won the pot uncontested.
                            Rewards aggressive pressure as a strategy, card-agnostic.

        Returns
        -------
        bonus   : total scalar bonus for this hand
        events  : dict of per-bonus fire counts for logging
        """
        bonus = 0.0
        events = {
            "elimination": 0,
            "survival": 0,
            "isolation": 0,
            "showdown": 0,
            "steal": 0,
        }

        if self._elimination_bonus > 0.0:
            for i, stack_before in stacks_before.items():
                if stack_before > 0 and self.agents[i].stack <= 0:
                    bonus += self._elimination_bonus
                    events["elimination"] += 1

        if self._survival_bonus > 0.0:
            players_with_chips = sum(1 for a in self.agents if a.stack > 0)
            if players_with_chips == 1 and self.agents[0].stack > 0:
                bonus += self._survival_bonus
                events["survival"] += 1

        if self._isolation_bonus > 0.0 and was_isolated:
            bonus += self._isolation_bonus
            events["isolation"] += 1

        if self._showdown_bonus > 0.0 and reached_showdown:
            bonus += self._showdown_bonus
            events["showdown"] += 1

        if self._steal_bonus > 0.0 and was_steal:
            bonus += self._steal_bonus
            events["steal"] += 1

        return bonus, events

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def play(self, total_hands: int) -> tuple[list, dict, dict]:
        """
        Run a full game session and return the collected trajectory,
        aggregated bonus event counts, and a log_data dict of metrics
        that callers can forward to a logging backend (e.g. wandb).

        Returns
        -------
        trajectory   : list of (PreprocessedObs, Action, reward) steps
        bonus_events : dict with total fire counts for each bonus type
        log_data     : dict of loggable scalar metrics from this game
        """
        self.reset()
        if self.table is None:
            raise Exception("Table should not be None")

        rewards_trajectories = []
        hands_trajectories = []

        total_bonus_events = {
            "elimination": 0,
            "survival": 0,
            "isolation": 0,
            "showdown": 0,
            "steal": 0,
        }

        for _ in range(total_hands):
            hand_trajectory = []

            players_with_chips = sum(
                1 for a in self.agents if a.stack >= MIN_STACK_TO_PLAY
            )
            if players_with_chips < 2:
                break

            stacks_before = self._stack_snapshot()

            # Only reset internal state for agents still in the game —
            # calling new_hand() on eliminated agents wastes allocations.
            for agent in self.agents:
                if agent.stack >= MIN_STACK_TO_PLAY:
                    agent.new_hand()

            obs_array = self.table.reset_hand()
            obs = Observation(
                obs_array,
                self._get_point_of_view(obs_array[0], self.table.hand_log),
            )

            rewards = np.zeros(len(self.agents))
            reached_showdown = False
            active_count = self.active_opponents + 1

            while True:
                acting_player_i = int(obs.player_identifier)

                if acting_player_i >= len(self.agents):
                    raise Exception(
                        f"player_identifier {acting_player_i} out of range "
                        f"(agents: {len(self.agents)})"
                    )

                acting_agent = self.agents[acting_player_i]

                if acting_agent.state is not PlayerState.ACTIVE:
                    print(
                        f"Table asked inactive player {acting_player_i} — forcing _end_hand"
                    )
                    self.table.hand_is_over = True
                    self.table._end_hand()
                    rewards = np.asarray(
                        [
                            p.get_reward()
                            for p in sorted(self.agents, key=lambda a: a.identifier)
                        ]
                    )
                    break

                action = acting_agent.get_action(obs)

                if acting_player_i == 0:
                    preprocessed = self.current_model.preprocess(obs)
                    hand_trajectory.append((preprocessed, action))

                obs_array, rewards, done = self.table.step(action)

                if done:
                    # Count players still in the hand at resolution —
                    # read before any further state mutation.
                    players_in_hand = sum(
                        1
                        for a in self.agents
                        if a.state in (PlayerState.ACTIVE, PlayerState.ALL_IN)
                    )
                    reached_showdown = players_in_hand > 1
                    active_count = players_in_hand
                    break

                obs = Observation(
                    obs_array,
                    self._get_point_of_view(obs_array[0], self.table.hand_log),
                )

            main_reward = rewards[0]
            if main_reward is not None:
                was_isolated = active_count == 2

                # Steal requires UGO to not be all-in: an all-in win where
                # opponents fold is a different outcome, not a positional steal.
                was_steal = (
                    not reached_showdown
                    and not self.agents[0].all_in
                    and float(main_reward) > 0
                )

                hand_bonus, hand_events = self._compute_hand_bonus(
                    stacks_before,
                    reached_showdown=reached_showdown,
                    was_isolated=was_isolated,
                    was_steal=was_steal,
                )

                for k, v in hand_events.items():
                    total_bonus_events[k] += v

                effective_reward = float(main_reward)
                if reached_showdown and self._showdown_reward_multiplier != 1.0:
                    effective_reward *= self._showdown_reward_multiplier
                rewards_trajectories.append(effective_reward + hand_bonus)
                hands_trajectories.append(hand_trajectory)

            if self.agents[0].stack <= 0:
                break

        gamma = self.config.get("decadiment_factor", 0.0)
        for hand_trajectory, reward in zip(
            hands_trajectories,
            self.get_weighted_rewards(rewards_trajectories, gamma),
        ):
            for item in hand_trajectory:
                self.trajectory.append((*item, reward))

        log_data = {
            "game/showdown_multiplier": self._showdown_reward_multiplier,
        }
        return self.trajectory, total_bonus_events, log_data

    # ------------------------------------------------------------------
    # Observation utility
    # ------------------------------------------------------------------

    def _get_point_of_view(self, player, hand_log):
        """
        Remap player identifiers in the hand log so that the acting player
        always sees itself as player 0. Preserves relative ordering.
        """
        if player == 0:
            return hand_log
        hand_log = hand_log.copy()
        mask = hand_log[:, 0] != -1.0
        hand_log[mask, 0] = (hand_log[mask, 0] - player) % (self.active_opponents + 1)
        return hand_log
