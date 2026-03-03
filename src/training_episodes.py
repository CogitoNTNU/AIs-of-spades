import random as rn
import PlayerAgent
from pokerenv.table import Table

class TrainingEpisode:
    def __init__(self):
        active_opponents = rn.randint(1, 5)
        opponents = [PlayerAgent(self.take_random_weights()) for _ in range(active_opponents)]
        us = PlayerAgent(self.poker_network)
        agents = [us] + opponents
        player_names = {0: 'TrackedAgent1'}

        track_single_player = True 

        # Bounds for randomizing player stack sizes in reset()
        low_stack_bbs = 50
        high_stack_bbs = 200
        hand_history_location = 'hands/'
        invalid_action_penalty = 0
        table = table(active_opponents + 1, 
                    player_names=player_names,
                    track_single_player=track_single_player,
                    stack_low=low_stack_bbs,
                    stack_high=high_stack_bbs,
                    hand_history_location=hand_history_location,
                    invalid_action_penalty=invalid_action_penalty
        )
        table.seed(1)


    def take_random_weights(self):
        pass