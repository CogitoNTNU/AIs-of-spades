import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action, action_list
import torch
import torch.nn as nn
import torch.optim as optim
from config import *

class LearningLoop():
    def __init__(self):
        self.optimizer = optim.Adam(
            list(PN.parameters()),
            lr=LEARNING_RATE
        )
    def start_learning():
        for epoch in range(EPOCHS):
            batch_trajectories = []
            batch_rewards = []
            for x in range(BATCH_SIZE):
                reset_env()
                episode=EpisodeLoop(HANDS_PER_EPISODE)
                batch_trajectories.append(episode.get_trajectories())
                batch_rewards.append(episode.get_rewards())

            reward=compute_reward(batch_trajectories,batch_rewards)
            result=gradient_decent(reward)
            if epoch % 1000 == 0: 
                save_to_file(result)


    def compute_reward():
        pass

    def save_to_file():
        pass

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


