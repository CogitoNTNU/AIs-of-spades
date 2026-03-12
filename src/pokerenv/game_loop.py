import random as rn

from pokerenv.player_agent import PlayerAgent
from pokerenv.table import Table
from pokerenv.observation import Observation
from pokerenv.weight_manager import WeightManager

MAIN_CHARACTER_NAME = "UGO"


class Game:
    def __init__(self, weight_manager: WeightManager, current_model):
        self.weight_manager = weight_manager
        self.current_model = current_model
        self.table = None
        self.agents = []

        # trajectory: list of (log_p_discrete, log_p_continuous) — both PyTorch tensors
        # Only steps where the main character acted are stored.
        self.trajectory = []
        self.reward = 0.0

    def reset(self):
        self.trajectory = []
        self.reward = 0.0

        self.active_opponents = rn.randint(1, 5)
        player_names = {i: "player_%d" % (i + 1) for i in range(6)}
        player_names[0] = MAIN_CHARACTER_NAME

        self.agents = [PlayerAgent(0, MAIN_CHARACTER_NAME, 0, self.current_model)]
        for n in range(1, self.active_opponents + 1):
            self.agents.append(
                PlayerAgent(
                    n, player_names[n], 0, self.weight_manager.sample_opponent()
                )
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

    def play(self, total_hands: int):
        self.reset()
        if self.table is None:
            raise Exception("Table should not be None")

        for hand_index in range(total_hands):
            obs_array = self.table.reset_hand()
            obs = Observation(
                obs_array, self._get_point_of_view(obs_array[0], self.table.hand_log)
            )

            while True:
                acting_player_i = int(obs.player_identifier)

                if acting_player_i >= len(self.agents):
                    raise Exception(
                        "player_identifier %d is out of range (agents: %d)"
                        % (acting_player_i, len(self.agents))
                    )

                action = self.agents[acting_player_i].get_action(obs)

                if acting_player_i == 0:
                    self.trajectory.append((obs, action))

                obs_array, rewards, done, _ = self.table.step(action)

                if done:
                    main_character = self.table.get_player_by_name(MAIN_CHARACTER_NAME)
                    if main_character is not None:
                        reward = main_character.get_reward()
                        if reward is not None:
                            self.reward += reward

                        if main_character.stack <= 0:
                            return self.reward, self.trajectory
                    break

                obs = Observation(
                    obs_array,
                    self._get_point_of_view(obs_array[0], self.table.hand_log),
                )

        return self.reward, self.trajectory

    def _get_point_of_view(self, player, hand_log):
        if player == 0:
            return hand_log
        hand_log = hand_log.copy()  # evita modifiche in-place
        mask = hand_log[:, 0] != -1.0
        hand_log[mask, 0] = (hand_log[mask, 0] - player) % (self.active_opponents + 1)
        return hand_log
