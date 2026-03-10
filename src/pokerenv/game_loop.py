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

        active_opponents = rn.randint(1, 5)
        player_names = {i: "player_%d" % (i + 1) for i in range(6)}
        player_names[0] = MAIN_CHARACTER_NAME

        self.agents = [PlayerAgent(0, MAIN_CHARACTER_NAME, 0, self.current_model)]
        for n in range(1, active_opponents + 1):
            self.agents.append(
                PlayerAgent(
                    n, player_names[n], 0, self.weight_manager.sample_opponent()
                )
            )

        self.table = Table(
            active_opponents + 1,
            players=self.agents,
            stack_low=50,
            stack_high=200,
            hand_history_location="hands/",
            invalid_action_penalty=0,
        )
        self.table.seed(None)

    def play(self, total_hands: int):
        if self.table is None:
            raise Exception("Table should not be None")

        self.reset()
        for hand_index in range(total_hands):
            obs_array = self.table.reset_hand()
            obs = Observation(obs_array)

            while True:
                acting_player_i = int(obs.player_identifier)

                if acting_player_i >= len(self.agents):
                    raise Exception(
                        "player_identifier %d is out of range (agents: %d)"
                        % (acting_player_i, len(self.agents))
                    )

                action = self.agents[acting_player_i].get_action(obs)

                if acting_player_i == 0:
                    self.trajectory.append(
                        (action.log_p_discrete, action.log_p_continuous)
                    )

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

                obs = Observation(obs_array)

        return self.reward, self.trajectory
