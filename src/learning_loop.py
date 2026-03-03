import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action, action_list
import torch
import torch.nn as nn
import torch.optim as optim
from config import *



def reset_env():
    PN.reset_trajectories() #We added
    batch_rewards = [] #We added (will not work)!!!
    active_players = 6
    agents = [ExampleRandomAgent() for _ in range(6)]
    player_names = {0: 'TrackedAgent1', 1: 'Agent2'} # Rest are defaulted to player3, player4...
    # Should we only log the 0th players (here TrackedAgent1) private cards to hand history files
    

for epoch in range(EPOCHS):
    batch_trajectories = []
    batch_rewards = []
    for  ex in range(BATCH_SIZE):
        reset_env()
        done = False

class LearningLoop():
    def __init__(self):
        self.optimizer = optim.Adam(
            list(PN.parameters()),
            lr=LEARNING_RATE
        )
    
    def compute_reward(self, final_stack, initial_stack=50):
        return np.log(final_stack/initial_stack)

if __name__ == "__main__":
    learning_loop = LearningLoop()












iteration = 1
while True:
    if iteration % 50 == 0:
        table.hand_history_enabled = True
    active_players = np.random.randint(2, 7)
    table.n_players = active_players
    obs = table.reset()
    for agent in agents:
        agent.reset()
    acting_player = int(obs[indices.ACTING_PLAYER])
    while True:
        action = agents[acting_player].get_action(obs)
        obs, reward, done, _ = table.step(action)
        if  done:
            # Distribute final rewards
            for i in range(active_players):
                agents[i].rewards.append(reward[i])
            break
        else:
            # This step can be skipped unless invalid action penalty is enabled, 
            # since we only get a reward when the pot is distributed, and the done flag is set
            agents[acting_player].rewards.append(reward[acting_player])
            acting_player = int(obs[indices.ACTING_PLAYER])
    iteration += 1
    table.hand_history_enabled = False


