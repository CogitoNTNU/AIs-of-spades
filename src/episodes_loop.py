import random as rn
import PlayerAgent
from pokerenv.table import Table
from pokerenv.observation import Observation
from weight_manager import WeightManager
import pokerenv.obs_indices as indices


MAIN_CHARACTER_NAME = "UGO"


class TrainingEpisode:
    def __init__(self):
        self.reset()
        self.trajectory = []
        self.reward = 0

    def reset(self):
        active_opponents = rn.randint(1, 5)
        opponents = [
            PlayerAgent(WeightManager.get_opponent_weights())
            for _ in range(active_opponents)
        ]
        main_character = PlayerAgent(WeightManager.get_updated_weights())
        self.agents = [main_character] + opponents

        player_names = {0: MAIN_CHARACTER_NAME}

        # Bounds for randomizing player stack sizes in reset()
        low_stack_bbs = 50
        high_stack_bbs = 200
        hand_history_location = "hands/"
        invalid_action_penalty = 0
        self.table = Table(
            active_opponents + 1,
            player_names=player_names,
            stack_low=low_stack_bbs,
            stack_high=high_stack_bbs,
            hand_history_location=hand_history_location,
            invalid_action_penalty=invalid_action_penalty,
        )
        self.table.seed(1)

    def play(self):
        self.reset()

        iteration = 1
        while True:
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
                else:
                    # This step can be skipped unless invalid action penalty is enabled,
                    # since we only get a reward when the pot is distributed, and the done flag is set
                    agents[acting_player].rewards.append(reward[acting_player])
                    acting_player = self.agents[obs.player_identifier]

