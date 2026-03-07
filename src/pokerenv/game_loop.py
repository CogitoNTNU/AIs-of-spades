import random as rn
from src.pokerenv.player_agent import PlayerAgent
from src.pokerenv.table import Table
from src.pokerenv.observation import Observation
from src.pokerenv.weight_manager import WeightManager
import src.pokerenv.obs_indices as indices

MAIN_CHARACTER_NAME = "UGO"

class Game:
    def __init__(self, weight_manager: WeightManager, current_model, ):
        self.weight_manager = weight_manager
        self.current_model = current_model

        self.trajectory = []
        self.reward = 0

    def reset(self):
        active_opponents = rn.randint(1, 5)
        player_names = {0: MAIN_CHARACTER_NAME}

        for player in range(6):
            if player not in player_names.keys():
                player_names[player] = "player_%d" % (player + 1)

        self.agents = [
            PlayerAgent(self.current_model, 0, player_names[0])
        ]

        for n in range(1, active_opponents + 1):
            self.agents.append(
                PlayerAgent(self.weight_manager.sample_opponent(), n, player_names[n])
            )

        # Bounds for randomizing player stack sizes in reset()
        low_stack_bbs = 50
        high_stack_bbs = 200
        hand_history_location = "hands/"
        invalid_action_penalty = 0
        self.table = Table(
            active_opponents + 1,
            players=self.agents,
            stack_low=low_stack_bbs,
            stack_high=high_stack_bbs,
            hand_history_location=hand_history_location,
            invalid_action_penalty=invalid_action_penalty,
        )
        self.table.seed(1)

    def play(self, total_iterations):
        self.reset()

        iteration = 1
        while iteration < total_iterations:
            obs = Observation(self.table.reset())
            acting_player = self.agents[obs.player_identifier]
            while True:
                action = self.agents[acting_player].get_action(obs)
                obs, reward, done, _ = self.table.step(action)
                self.trajectory.append(
                    (action.action_probability, action.bet_probability)
                )
                if done:
                    main_character = self.table.get_player_by_name(MAIN_CHARACTER_NAME)
                    if main_character:
                        reward = main_character.get_reward()
                        if reward:
                            self.reward += reward
                    break

        return self.reward
