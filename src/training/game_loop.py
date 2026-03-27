import random as rn
import numpy as np

from pokerenv.table import Table
from pokerenv.observation import Observation
from pokerenv.common import PlayerState

from training.player_agent import PlayerAgent

MAIN_CHARACTER_NAME = "UGO"


class Game:

    def __init__(self, opponents: list, current_model, config):
        self.opponents = opponents  # list of 5 pre-built models
        self.current_model = current_model
        self.table = None
        self.agents = []
        self.config = config

        # trajectory: list of (PreprocessedObs, Action)
        # Only steps where the main character acted are stored.
        # Preprocessing is done here in the worker to avoid redundant work
        # on the main process.
        self.trajectory = []
        self.reward = 0.0

    def reset(self):
        self.trajectory = []
        self.hand_rewards = []
        self.reward = 0.0

        self.active_opponents = rn.randint(1, 5)
        player_names = {i: "player_%d" % (i + 1) for i in range(6)}
        player_names[0] = MAIN_CHARACTER_NAME

        self.agents = [PlayerAgent(0, MAIN_CHARACTER_NAME, 0, self.current_model)]
        for n in range(1, self.active_opponents + 1):
            self.agents.append(
                PlayerAgent(n, player_names[n], 0, self.opponents[n - 1])
            )

        self.table = Table(
            self.active_opponents + 1,
            players=self.agents,
            stack_low=50,
            stack_high=200,
            hand_history_location="hands/",
            invalid_action_penalty=0,
        )
        self.table.seed(None)
        self.table.reset()

    def get_weighted_rewards(self, hand_rewards: list, gamma: float = 0.8) -> list:
        if len(hand_rewards) == 0:
            return []
        if len(hand_rewards) == 1:
            return hand_rewards
        weighted_rewards = self.get_weighted_rewards(hand_rewards[1:], gamma)
        return [hand_rewards[0] + gamma * weighted_rewards[0]] + weighted_rewards

    def play(self, total_hands: int):
        self.reset()
        if self.table is None:
            raise Exception("Table should not be None")

        rewards_trajectories = []
        hands_trajectories = []

        for hand_index in range(total_hands):
            hand_trajectory = []
            # End the game early if fewer than 2 players have chips to play.
            players_with_chips = sum(1 for a in self.agents if a.stack >= 1)
            if players_with_chips < 2:
                break

            for agent in self.agents:
                agent.new_hand()

            obs_array = self.table.reset_hand()
            obs = Observation(
                obs_array,
                self._get_point_of_view(obs_array[0], self.table.hand_log),
            )

            rewards = np.zeros(len(self.agents))

            while True:
                acting_player_i = int(obs.player_identifier)

                if acting_player_i >= len(self.agents):
                    raise Exception(
                        "player_identifier %d is out of range (agents: %d)"
                        % (acting_player_i, len(self.agents))
                    )

                acting_agent = self.agents[acting_player_i]

                if acting_agent.state != PlayerState.ACTIVE or acting_agent.all_in:
                    print("Table asked inactive player %d — forcing _end_hand")
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
                    # Preprocess here in the worker (CPU, parallel) so the main
                    # process only needs to stack tensors and run the GPU forward.
                    preprocessed = self.current_model.preprocess(obs)
                    hand_trajectory.append((preprocessed, action))

                obs_array, rewards, done = self.table.step(action)

                if done:
                    break

                obs = Observation(
                    obs_array,
                    self._get_point_of_view(obs_array[0], self.table.hand_log),
                )

            main_reward = rewards[0]
            if main_reward is not None:
                rewards_trajectories.append(main_reward)
                hands_trajectories.append(hand_trajectory)

            if self.agents[0].stack <= 0:
                break

        for hand_trajectory, reward in zip(
            hands_trajectories,
            self.get_weighted_rewards(
                rewards_trajectories, self.config.get("decadiment_factor", 0.0)
            ),
        ):
            for item in hand_trajectory:
                self.trajectory.append((*item, reward))

        return self.trajectory

    def _get_point_of_view(self, player, hand_log):
        if player == 0:
            return hand_log
        hand_log = hand_log.copy()
        mask = hand_log[:, 0] != -1.0
        hand_log[mask, 0] = (hand_log[mask, 0] - player) % (self.active_opponents + 1)
        return hand_log
